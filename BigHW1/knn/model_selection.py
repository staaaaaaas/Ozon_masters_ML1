from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif isinstance(cv, int):
        cv = KFold(n_splits=cv)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))

    cv_results = defaultdict(lambda: np.zeros(cv.get_n_splits(X)))
    max_k = np.max(k_list)
    for i, split in enumerate(cv.split(X)):
        train, test = split
        model = BatchedKNNClassifier(n_neighbors=max_k, **kwargs)

        model.fit(X[train], y[train])
        dists, inds = model.kneighbors(X[test], return_distance=True)
        for k in k_list:
            y_pred = model._predict_precomputed(inds[:, :k], dists[:, :k])
            cv_results[k][i] = scorer(y[test], y_pred)
    return cv_results
