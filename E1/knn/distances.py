import numpy as np


def euclidean_distance(x, y):
    xx = np.broadcast_to(np.sum(x ** 2, axis=1)[:, np.newaxis], (x.shape[0],
                                                                 y.shape[0]))
    yy = np.broadcast_to(np.sum(y ** 2, axis=1)[np.newaxis], (x.shape[0],
                                                              y.shape[0]))
    xy = x @ y.T
    return np.sqrt(xx - 2 * xy + yy)


def cosine_distance(x, y):
    xx = np.broadcast_to(np.sum(x ** 2, axis=1)[:, np.newaxis], (x.shape[0],
                                                                 y.shape[0]))
    yy = np.broadcast_to(np.sum(y ** 2, axis=1)[np.newaxis], (x.shape[0],
                                                              y.shape[0]))
    xy = x @ y.T
    return 1 - xy / np.sqrt(xx * yy)
