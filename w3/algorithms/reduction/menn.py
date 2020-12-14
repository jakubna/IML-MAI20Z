import numpy as np
from algorithms.KNNAlgorithm import KNNLAlgorithm


def enn_reduction(knn: KNNLAlgorithm, X: np.ndarray, y: np.ndarray):
    knn.fit(X, y)
    remove_intances = []
    for i in range(X.shape[0]):
        dist, kn=knn.kneighbors(X[i,:])
        max_distance = max(dist)
        kn=list(kn)
        for j in range(X.shape[0):
          if j != i:
              one_dist=knn._calculate_distance(X[i,:],X[j,:])
          if one_dist<=max_dist:
              kn.append(X[j,:])
        predxi = knn.predict(X[i, :])
        pred_neighbors = knn.predict(np.array(kn))
        if len(pred_neighbors.unique()) != 1:
            remove_intances.append(i)
        else:
            if pred_neighbors.unique() != predxi.unique()
                remove_intances.append(i)
    if len(remove_intances):
        # Remove instances from original data
        X = np.delete(X, remove_intances, axis=0)
        y = np.delete(y, remove_intances)
        
    return X, y
