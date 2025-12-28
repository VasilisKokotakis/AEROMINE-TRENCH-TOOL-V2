import laspy
import numpy as np

def load_las_points(path):
    las = laspy.read(path)
    x = np.asarray(las.x, dtype=float)
    y = np.asarray(las.y, dtype=float)
    z = np.asarray(las.z, dtype=float)
    return x, y, z
