import numpy as np


def calc_expectations(h, w, X, Q):
    R = np.apply_along_axis(lambda x: np.concatenate(
        [np.cumsum(x[:w - 1]), np.convolve(x, np.ones(w), 'valid')]), axis=1, arr=Q)
    R = np.apply_along_axis(lambda x: np.concatenate(
        [np.cumsum(x[:h - 1]), np.convolve(x, np.ones(h), 'valid')]), axis=0, arr=R)
    return X * R
