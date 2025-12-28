import os
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# χρησιμοποιεί τον δικό σας κώδικα
from sections.io import load_las_points
from sections.processing import auto_axis, compute_sections

BASE = Path(__file__).parent.resolve()
RUNS = BASE / "runs"
RUNS.mkdir(exist_ok=True)

app = FastAPI()

# CORS για να μπορεί το index.html να μιλάει στον server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("section_id", as_index=False)
        .agg(
            count=("z", "size"),
            z_min=("z", "min"),
            z_max=("z", "max"),
            x_min=("dist_off", "min"),
            x_max=("dist_off", "max"),
        )
    )
    # Filter out sections with too few points
    summary = summary[summary["count"] >= 1000].copy()
    # Add wall distance and depth calculations
    summary["wall_distance"] = summary["x_max"] - summary["x_min"]
    summary["depth"] = summary["z_max"] - summary["z_min"]
    return summary


def auto_edges_from_section(
    df_sec: pd.DataFrame,
    bin_w: float = 0.02,
    q: float = 0.95,
    smooth_bins: int = 7,
    slope_thr: float = 1.5,
    margin: float = 0.02,
):
    """Return (left_edge, right_edge) from dist_off vs z profile.
    Uses an upper-quantile envelope + smoothing + slope thresholding.
    Returns (None, None) if edges cannot be detected.
    """
    if df_sec.empty:
        return None, None

    x = df_sec["dist_off"].to_numpy()
    z = df_sec["z"].to_numpy()

    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        return None, None

    bins = np.arange(xmin, xmax + bin_w, bin_w)
    if len(bins) < 10:
        return None, None

    dfb = pd.DataFrame({"x": x, "z": z})
    dfb["bin"] = pd.cut(dfb["x"], bins=bins, include_lowest=True)

    env = dfb.groupby("bin")["z"].quantile(q).reset_index()
    env["xc"] = env["bin"].apply(lambda b: (b.left + b.right) / 2.0)
    env = env.dropna()
    if len(env) < 10:
        return None, None

    env["zs"] = env["z"].rolling(smooth_bins, center=True, min_periods=1).mean()

    xc = env["xc"].to_numpy()
    zs = env["zs"].to_numpy()
    if len(xc) < 5:
        return None, None

    slope = np.gradient(zs, xc)

    idx = np.where(np.abs(slope) >= slope_thr)[0]
    if len(idx) == 0:
        return None, None

    left_edge = float(xc[idx[0]] - margin)
    right_edge = float(xc[idx[-1]] + margin)
    return left_edge, right_edge


@app.post("/run")
async def run(
    file: UploadFile,
    spacing: float = Form(0.10),
    prefilter_half_width: float = Form(2.0),
    edgelock: float = Form(0.05),
    clip_mode: str = Form("fixed"),  # "fixed" ή "auto"
    half_width: float = Form(0.7),  # για fixed ή fallback
    right_trim: float = Form(0.0),  # επιπλέον κόψιμο από δεξιά
    slope_thr: float = Form(1.5),  # για auto edges
    autoaxis: str = Form("1"),
):
    run_id = uuid.uuid4().hex[:10]
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    in_path = run_dir / file.filename
    with open(in_path, "wb") as f:
        f.write(await file.read())

    logs: list[str] = []
    logs.append(f"[input] {in_path.name}")

    # load
    x, y, z = load_las_points(str(in_path))
    logs.append(f"[loaded] points={len(x):,}")

    # axis
    if autoaxis == "1":
        start, end = auto_axis(x, y)
        logs.append("[axis] auto (PCA)")
    else:
        start = np.array([x.min(), y.min()], dtype=float)
        end = np.array([x.max(), y.max()], dtype=float)
        logs.append("[axis] bbox diagonal")

    # compute sections
    df = compute_sections(
        x,
        y,
        z,
        start,
        end,
        spacing,
        prefilter_half_width=prefilter_half_width,
        edge_lock_margin=edgelock,
    )

    logs.append(f"[sections] rows={len(df):,} sections={df['section_id'].nunique()}")

    # FINAL CLIP: fixed or auto
    mode = (clip_mode or "fixed").strip().lower()
    if mode == "fixed":
        df = df[df["dist_off"].between(-half_width, half_width - right_trim)].copy()
        logs.append(f"[clip:fixed] half_width={half_width} right_trim={right_trim}")
    else:
        out = []
        fail = 0
        for sid, g in df.groupby("section_id"):
            le, re = auto_edges_from_section(g, slope_thr=slope_thr)
            if le is None or re is None or le >= re:
                fail += 1
                gg = g[g["dist_off"].between(-half_width, half_width)].copy()
                out.append(gg)
                continue

            gg = g[(g["dist_off"] >= le) & (g["dist_off"] <= re)].copy()
            out.append(gg)

        df = pd.concat(out, ignore_index=True) if out else df.iloc[0:0].copy()
        logs.append(f"[clip:auto] slope_thr={slope_thr} failed_sections={fail}")
        if len(df) > 0:
            logs.append(
                f"[clip:auto] dist_off=[{df['dist_off'].min():.3f},{df['dist_off'].max():.3f}]"
            )

    # save outputs
    full_csv = run_dir / "sections.csv"
    summary_csv = run_dir / "sections_summary.csv"

    df.to_csv(full_csv, index=False)
    summary = build_summary(df)
    summary.to_csv(summary_csv, index=False)

    # Add wall distance statistics to logs
    if len(summary) > 0:
        wall_distances = summary["wall_distance"]
        depths = summary["depth"]
        logs.append(f"[walls] distance_avg={wall_distances.mean():.3f}m min={wall_distances.min():.3f}m max={wall_distances.max():.3f}m")
        logs.append(f"[depth] avg={depths.mean():.3f}m min={depths.min():.3f}m max={depths.max():.3f}m")

    return JSONResponse(
        {
            "ok": True,
            "run_id": run_id,
            "full_csv": f"{run_id}/sections.csv",
            "summary_csv": f"{run_id}/sections_summary.csv",
            "log": "\n".join(logs),
        }
    )


@app.get("/download/{path:path}")
def download(path: str):
    p = RUNS / path
    if not p.exists():
        return JSONResponse({"ok": False, "error": "File not found"}, status_code=404)
    return FileResponse(str(p), filename=p.name)

@app.get("/")
def read_root():
    return FileResponse("index.html", media_type="text/html")
