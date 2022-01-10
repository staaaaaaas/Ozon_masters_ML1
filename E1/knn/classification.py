import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    EPS = 1e-5

    def __init__(
            self,
            n_neighbors,
            algorithm='my_own',
            metric='euclidean',
            weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(
                n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        if self._weights == 'uniform':
            y_pred = np.apply_along_axis(
                lambda ind: np.argmax(
                    np.bincount(
                        self._labels[ind])),
                axis=1,
                arr=indices)
        elif self._weights == 'distance':
            W = 1 / (distances + self.EPS)
            weights_iter = np.arange(indices.shape[0])[:, np.newaxis]
            indices = np.concatenate((indices, weights_iter), axis=1)
            y_pred = np.apply_along_axis(
                lambda ind: np.argmax(
                    np.bincount(
                        self._labels[ind[:-1]],
                        weights=W[ind[-1]])),
                axis=1,
                arr=indices)
        return y_pred

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedKNNClassifier(KNNClassifier):

    def __init__(
            self,
            n_neighbors,
            algorithm='my_own',
            metric='euclidean',
            weights='uniform',
            batch_size=None):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights,
            metric=metric,
        )
        self._batch_size = batch_size

    def kneighbors(self, X, return_distance=False):
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=return_distance)
        if return_distance:
            res_distances = np.zeros((X.shape[0], self._finder.n_neighbors))
        res_indices = np.zeros(
            (X.shape[0], self._finder.n_neighbors), dtype='int64')
        for ind in range(0, X.shape[0], self._batch_size):
            next_ind = ind + self._batch_size
            if return_distance:
                distances, indices = super().kneighbors(
                    X[ind: next_ind], return_distance=return_distance)
                res_distances[ind: next_ind] = distances
            else:
                indices = super().kneighbors(
                    X[ind: next_ind], return_distance=return_distance)
            res_indices[ind: next_ind] = indices
        if return_distance:
            return res_distances, res_indices
        return res_indices
