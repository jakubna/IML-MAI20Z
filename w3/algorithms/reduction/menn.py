import numpy as np
from algorithms.kNNAlgorithm import kNNAlgorithm


def menn_reduction(knn: kNNAlgorithm, X: np.ndarray, y: np.ndarray):
    knn.fit(X, y)
    remove_intances = []
    for i in range(X.shape[0]):
        dist, kn = knn.kneighbors(np.array([X[i, :]]),return_distance=True)
        max_distance = np.max(dist)
        kn = kn.tolist()
        kn=kn[0]
        for j in range(X.shape[0]):
            if j != i:
                one_dist = knn._calculate_distance(np.array([X[i, :]]), np.array([X[j, :]]))
                if one_dist[0][0] <= max_distance:
                    kn.append(j)
        predxi = knn.predict(np.array([X[i, :]]))
        pred_neighbors = knn.predict(X[kn,:])
        if len(set(pred_neighbors)) != 1:
            remove_intances.append(i)
        else:
            if set(pred_neighbors) != set(predxi):
                remove_intances.append(i)
    if len(remove_intances):
        # Remove instances from original data
        X = np.delete(X, remove_intances, axis=0)
        y = np.delete(y, remove_intances)
        
    return X, y
