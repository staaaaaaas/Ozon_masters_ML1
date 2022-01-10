import numpy as np


def encode_rle(x):
    change_ind = np.where(x[:-1] != x[1:])[0] + 1
    change_ind = np.concatenate([[0], change_ind])
    nums = x[change_ind]
    counts = np.diff(np.concatenate([change_ind, [x.shape[0]]]))
    return nums, counts
