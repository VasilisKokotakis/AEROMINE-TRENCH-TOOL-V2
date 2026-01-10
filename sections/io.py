from __future__ import annotations

from typing import Tuple

import numpy as np


def load_las_points(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, z arrays (float) from a LAS/LAZ file."""
    import laspy

    las = laspy.read(path)
    x = np.asarray(las.x, dtype=float)
    y = np.asarray(las.y, dtype=float)
    z = np.asarray(las.z, dtype=float)
    return x, y, z
