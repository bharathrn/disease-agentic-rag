import numpy as np

def norm_vec(v: np.ndarray):
    """Normalize a vector."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
