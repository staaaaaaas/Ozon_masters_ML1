import numpy as np


def get_max_after_zero(x):
    if np.all(x != 0):
        return None
    zeroes_ind = np.where(x == 0)[0]
    zeroes_ind = zeroes_ind[zeroes_ind != x.shape[0] - 1]
    if zeroes_ind.shape[0] == 0:
        return None
    return np.max(x[zeroes_ind + 1])
