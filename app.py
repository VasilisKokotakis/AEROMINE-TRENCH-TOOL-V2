import os
import uuid
import zipfile
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

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


# Analysis functions
def create_cross_section_plots(df_sections, out_dir, point_size=1):
    """Create individual cross-section plots"""
    import matplotlib.pyplot as plt

    section_ids = sorted(df_sections["section_id"].unique())
    print(f"Found {len(section_ids)} sections: {section_ids[0]} to {section_ids[-1]}")

    for sid in section_ids:
        sub = df_sections[df_sections["section_id"] == sid]

        plt.figure(figsize=(12, 8))
        plt.scatter(sub["dist_off"], sub["z"], s=point_size, alpha=0.6, c='blue', edgecolors='none')
        plt.xlabel("Distance from trench axis (m)")
        plt.ylabel("Elevation (m)")
        plt.title(f"Cross-section {sid}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(out_dir / f"cross_section_{sid:02d}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✅ Saved {len(section_ids)} cross-section plots")


def create_summary_analysis_plots(df_summary, out_dir):
    """Create summary analysis plots"""
    import matplotlib.pyplot as plt

    df = df_summary.sort_values("section_id")

    # Wall distance profile
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(df["section_id"], df["wall_distance"], 'b-', linewidth=2, marker='o', markersize=3)
    plt.xlabel("Section ID")
    plt.ylabel("Wall Distance (m)")
    plt.title("Trench Wall Distance Along Length")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=df["wall_distance"].mean(), color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {df["wall_distance"].mean():.3f}m')
    plt.legend()

    # Depth profile
    plt.subplot(2, 1, 2)
    plt.plot(df["section_id"], df["depth"], 'g-', linewidth=2, marker='s', markersize=3)
    plt.xlabel("Section ID")
    plt.ylabel("Depth (m)")
    plt.title("Trench Depth Along Length")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=df["depth"].mean(), color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {df["depth"].mean():.3f}m')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "trench_profile_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Combined profile
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax1.plot(df["section_id"], df["wall_distance"], 'b-', linewidth=2, marker='o', markersize=4, label='Wall Distance')
    ax1.set_xlabel("Section ID")
    ax1.set_ylabel("Wall Distance (m)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["section_id"], df["depth"], 'g-', linewidth=2, marker='s', markersize=4, label='Depth')
    ax2.set_ylabel("Depth (m)", color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    plt.title("Combined Trench Profile Analysis")
    fig.tight_layout()
    plt.savefig(out_dir / "combined_profile_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Statistics summary
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Wall distance histogram
    ax1.hist(df["wall_distance"], bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel("Wall Distance (m)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Wall Distance Distribution")
    ax1.grid(True, alpha=0.3)

    # Depth histogram
    ax2.hist(df["depth"], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel("Depth (m)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Depth Distribution")
    ax2.grid(True, alpha=0.3)

    # Wall distance vs Depth scatter
    ax3.scatter(df["wall_distance"], df["depth"], alpha=0.6, color='red', s=50)
    ax3.set_xlabel("Wall Distance (m)")
    ax3.set_ylabel("Depth (m)")
    ax3.set_title("Wall Distance vs Depth Correlation")
    ax3.grid(True, alpha=0.3)

    # Section count bar chart
    ax4.bar(df["section_id"], df["count"], alpha=0.7, color='orange', width=0.8)
    ax4.set_xlabel("Section ID")
    ax4.set_ylabel("Point Count")
    ax4.set_title("Points per Section")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "statistics_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Saved summary analysis plots")


def generate_report(df_summary, out_dir):
    """Generate text report"""
    df = df_summary.sort_values("section_id")

    with open(out_dir / "complete_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write("AEROMINE TRENCH ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("SUMMARY STATISTICS:\n")
        f.write(f"Total sections: {len(df)}\n")
        f.write(f"Section ID range: {df['section_id'].min()} to {df['section_id'].max()}\n\n")

        f.write("WALL DISTANCE:\n")
        f.write(f"  Mean: {df['wall_distance'].mean():.3f}m\n")
        f.write(f"  Min:  {df['wall_distance'].min():.3f}m\n")
        f.write(f"  Max:  {df['wall_distance'].max():.3f}m\n")
        f.write(f"  Std:  {df['wall_distance'].std():.3f}m\n\n")

        f.write("DEPTH:\n")
        f.write(f"  Mean: {df['depth'].mean():.3f}m\n")
        f.write(f"  Min:  {df['depth'].min():.3f}m\n")
        f.write(f"  Max:  {df['depth'].max():.3f}m\n")
        f.write(f"  Std:  {df['depth'].std():.3f}m\n\n")

        f.write("VARIABILITY:\n")
        f.write(f"  Wall distance CV: {df['wall_distance'].std() / df['wall_distance'].mean() * 100:.1f}%\n")
        f.write(f"  Depth CV: {df['depth'].std() / df['depth'].mean() * 100:.1f}%\n\n")

        f.write("SECTION DETAILS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Section':<8} {'Wall Dist':<10} {'Depth':<8} {'Points':<8} {'Z Range':<15}\n")
        f.write("-" * 80 + "\n")

        for _, row in df.iterrows():
            z_range = f"{row['z_min']:.2f}-{row['z_max']:.2f}"
            f.write(f"{row['section_id']:<8.0f} {row['wall_distance']:<10.3f} {row['depth']:<8.3f} {row['count']:<8} {z_range:<15}\n")

    print("📄 Detailed report saved")


def run_complete_analysis(sections_df, summary_df, run_dir):
    """Run complete analysis and create ZIP file"""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    out_dir = run_dir / "complete_analysis"
    out_dir.mkdir(exist_ok=True)

    # Run analysis
    create_cross_section_plots(sections_df, out_dir, point_size=1)
    create_summary_analysis_plots(summary_df, out_dir)
    generate_report(summary_df, out_dir)

    # Create ZIP file
    zip_path = run_dir / "complete_analysis.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in out_dir.rglob('*'):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(run_dir))

    return zip_path

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


@app.post("/analyze/{run_id}")
async def analyze_run(run_id: str):
    """Run complete analysis on existing CSV files from a run"""
    run_dir = RUNS / run_id
    if not run_dir.exists():
        return JSONResponse({"ok": False, "error": "Run not found"}, status_code=404)

    sections_csv = run_dir / "sections.csv"
    summary_csv = run_dir / "sections_summary.csv"

    if not sections_csv.exists() or not summary_csv.exists():
        return JSONResponse({"ok": False, "error": "CSV files not found"}, status_code=404)

    try:
        # Load the CSV files
        sections_df = pd.read_csv(sections_csv)
        summary_df = pd.read_csv(summary_csv)

        # Run analysis
        zip_path = run_complete_analysis(sections_df, summary_df, run_dir)

        return JSONResponse({
            "ok": True,
            "analysis_zip": f"{run_id}/complete_analysis.zip",
            "message": "Analysis completed successfully"
        })

    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": f"Analysis failed: {str(e)}"},
            status_code=500
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
