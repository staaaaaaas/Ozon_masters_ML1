import numpy as np


def replace_nan_to_means(X):
    mask = np.all(np.isnan(X), axis=0)
    X_new = X.copy()
    X_new[:, mask] = 0
    X_new = np.nan_to_num(X_new, copy=False, nan=np.nanmean(X_new, axis=0))
    return X_new
