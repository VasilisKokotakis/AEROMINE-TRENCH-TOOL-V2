from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

def validate_params(
    spacing: float,
    prefilter_half_width: float,
    edgelock: float,
    half_width: float,
    right_trim: float,
    slope_thr: float,
    depth_min: float,
    clip_mode: str,
) -> list[str]:
    """Validate processing parameters. Returns a list of error messages (empty = valid)."""
    errors = []
    if spacing <= 0:
        errors.append("Spacing must be > 0")
    if prefilter_half_width <= 0:
        errors.append("Prefilter half-width must be > 0")
    if edgelock < 0:
        errors.append("Edge-lock must be >= 0")
    if half_width <= 0:
        errors.append("Half-width must be > 0")
    if right_trim < 0:
        errors.append("Right trim must be >= 0")
    if slope_thr <= 0:
        errors.append("Slope threshold must be > 0")
    if depth_min < 0:
        errors.append("Depth min must be >= 0")
    if clip_mode not in ("fixed", "auto"):
        errors.append("Clip mode must be 'fixed' or 'auto'")
    return errors


def auto_axis(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Detect the main axis direction using PCA on (x,y)."""
    pts = np.column_stack([x, y]).astype(float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        raise ValueError('Not enough points for axis detection')
    c = pts.mean(axis=0)
    P = pts - c
    cov = (P.T @ P) / max(1, (len(P)-1))
    w, v = np.linalg.eigh(cov)
    d = v[:, np.argmax(w)]
    d = d / (np.linalg.norm(d) + 1e-12)
    t = P @ d
    tmin, tmax = float(np.min(t)), float(np.max(t))
    start = c + d * tmin
    end = c + d * tmax
    return start.astype(float), end.astype(float)

def compute_sections(x: np.ndarray, y: np.ndarray, z: np.ndarray, start: np.ndarray, end: np.ndarray, spacing: float, prefilter_half_width: float = 0.5, edge_lock_margin: float = 0.05) -> pd.DataFrame:
    """Project points to the trench axis and assign section ids."""
    if spacing <= 0:
        raise ValueError('spacing must be > 0')
    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(end, dtype=float)
    u = p1 - p0
    L = float(np.linalg.norm(u))
    if not np.isfinite(L) or L <= 0:
        raise ValueError('Invalid axis length')
    u = u / L
    v = np.array([-u[1], u[0]], dtype=float)
    pts = np.column_stack([x, y]).astype(float)
    p = pts - p0
    s = p @ u
    d = p @ v
    # prefilter near the axis
    m = np.isfinite(s) & np.isfinite(d) & np.isfinite(z)
    if prefilter_half_width is not None and prefilter_half_width > 0:
        m = m & (np.abs(d) <= (prefilter_half_width + float(edge_lock_margin)))
    s = s[m]
    d = d[m]
    zf = np.asarray(z, dtype=float)[m]
    # section ids (start at 0)
    s0 = float(np.min(s))
    sec_id = np.floor((s - s0) / float(spacing)).astype(int)
    df = pd.DataFrame({'section_id': sec_id, 's': s, 'dist_off': d, 'z': zf})
    return df
