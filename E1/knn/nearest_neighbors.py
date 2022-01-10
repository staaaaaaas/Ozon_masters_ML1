import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    best_ind = np.argpartition(ranks, top - 1, axis=axis)[:, :top]
    best_ind_tmp = np.argsort(
        np.take_along_axis(
            ranks,
            best_ind,
            axis=axis),
        axis=axis)
    best_ind = np.take_along_axis(best_ind, best_ind_tmp, axis=axis)

    if return_ranks:
        return np.take_along_axis(ranks, best_ind, axis=axis), best_ind
    else:
        return best_ind


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        dist = self._metric_func(X, self._X)
        return get_best_ranks(
            dist,
            self.n_neighbors,
            axis=1,
            return_ranks=return_distance)
