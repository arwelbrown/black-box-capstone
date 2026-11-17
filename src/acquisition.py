import numpy as np


def minmax_scale(values):
    values = np.asarray(values, dtype=float)

    low = np.min(values)
    high = np.max(values)

    if high - low < 1e-12:
        return np.zeros_like(values)

    return (values - low) / (high - low)

