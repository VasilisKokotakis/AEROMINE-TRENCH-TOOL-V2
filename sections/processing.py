import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def auto_axis(x, y):
    xy = np.vstack((x, y)).T
    pca = PCA(n_components=2)
    pca.fit(xy)
    v = pca.components_[0] / np.linalg.norm(pca.components_[0])
    ctr = xy.mean(axis=0)
    start = ctr - v * 10.0
    end   = ctr + v * 10.0
    print(f"[auto-axis] start=({start[0]}, {start[1]}), end=({end[0]}, {end[1]})")
    return start, end

def compute_sections(
    x, y, z,
    start, end,
    spacing,
    prefilter_half_width=None,
    edge_lock_margin=0.0
):
    """
    Sections πάνω στον άξονα start->end.
    - prefilter_half_width: κρατά μόνο σημεία με |dist_off| <= half_width (π.χ. 2.0m)
    - edge_lock_margin: κλειδώνει τα όρια κάθε τομής προς τα μέσα κατά margin (π.χ. 0.05m)
    """
    axis_vec = end - start
    axis_vec = axis_vec / np.linalg.norm(axis_vec)

    rel = np.vstack((x - start[0], y - start[1])).T
    along = rel @ axis_vec
    off = rel @ np.array([-axis_vec[1], axis_vec[0]])

    section_id = np.floor(along / spacing).astype(int)

    df = pd.DataFrame(
        {"x": x, "y": y, "z": z,
         "dist_along": along, "dist_off": off,
         "section_id": section_id}
    )

    # (A) Prefilter: κράτα μόνο λωρίδα κοντά στο χαντάκι
    if prefilter_half_width is not None:
        hw = float(prefilter_half_width)
        df = df[df["dist_off"].abs() <= hw].copy()

    # (B) Edge-lock: κόψε outliers 5cm μέσα από min/max για κάθε section
    m = float(edge_lock_margin or 0.0)
    if m > 0.0 and len(df) > 0:
        gmin = df.groupby("section_id")["dist_off"].transform("min")
        gmax = df.groupby("section_id")["dist_off"].transform("max")

        low = gmin + m
        high = gmax - m

        # Αν σε κάποιο section high<=low, μην το κόβεις (λίγα/κακά σημεία)
        ok = high > low
        mask = (~ok) | ((df["dist_off"] >= low) & (df["dist_off"] <= high))
        df = df[mask].copy()

    return df.reset_index(drop=True)

