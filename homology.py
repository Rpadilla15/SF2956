import numpy as np

def safe_linspace(diagram, num=100, default_max=1.0):
    """Return a safe linspace for t values based on finite deaths."""
    if diagram.size == 0:
        return np.linspace(0, default_max, num)
    finite_mask = np.isfinite(diagram[:, 1])
    if not finite_mask.any():
        return np.linspace(0, default_max, num)
    max_val = np.max(diagram[finite_mask][:, 1])
    if max_val == 0 or not np.isfinite(max_val):
        max_val = default_max
    return np.linspace(0, max_val, num)


def stable_rank(diagram, t_values, include_infinite=True):
    """Compute stable rank curve (count of intervals with lifetime > 2t)."""
    if diagram.size == 0:
        return np.zeros_like(t_values)

    finite_births = np.isfinite(diagram[:, 0])
    if include_infinite:
        dgm = diagram[finite_births]
    else:
        finite_deaths = np.isfinite(diagram[:, 1])
        dgm = diagram[finite_births & finite_deaths]

    if dgm.size == 0:
        return np.zeros_like(t_values)

    births, deaths = dgm[:, 0], dgm[:, 1]
    lifetimes = deaths - births
    lifetimes = np.nan_to_num(lifetimes, posinf=np.inf)

    return np.array([np.sum(lifetimes > 2 * t) for t in t_values])


def betti_curve(diagram, t_values):
    """Compute Betti curve: number of features alive at each t."""
    if diagram.size == 0:
        return np.zeros_like(t_values, dtype=int)

    finite_births = np.isfinite(diagram[:, 0])
    dgm = diagram[finite_births]
    if dgm.size == 0:
        return np.zeros_like(t_values, dtype=int)

    births, deaths = dgm[:, 0], dgm[:, 1]
    return np.array([np.sum((births <= t) & (t < deaths)) for t in t_values], dtype=int)


def safe_interp(x_new, x_orig, y_orig):
    """Interpolate safely, handling constant y_orig arrays."""
    if np.all(y_orig == y_orig[0]):
        return np.full_like(x_new, y_orig[0])
    return np.interp(x_new, x_orig, y_orig)