import numpy as np


def get_nonzero_diag_product(X):
    d = np.diag(X)
    if any(d != 0):
        return np.prod(d[d != 0])
    else:
        return None
