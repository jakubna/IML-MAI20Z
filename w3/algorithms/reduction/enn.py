import numpy as np
from algorithms.kNNAlgorithm import kNNAlgorithm


def enn_reduction(knn: kNNAlgorithm, X: np.ndarray, y: np.ndarray):
    knn.fit(X, y)
    remove_intances = []
    for i in range(X.shape[0]):
        pred = knn.predict(np.array([X[i, :]]))
        if pred[0] != y[i]:
            remove_intances.append(i)
    if len(remove_intances):
        # Remove instances from original data
        X = np.delete(X, remove_intances, axis=0)
        y = np.delete(y, remove_intances)
        
    return X, y
