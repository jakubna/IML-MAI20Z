import numpy as np
from algorithms.KNNAlgorithm import KNNAlgorithm


def fcnn_reduction(knn: KNNAlgorithm, X: np.ndarray, y: np.ndarray):
    knn.fit(X,y)
    S, V = [], []
    

    #initialization with centroids of each different class
    for label in set(y):
        i_labels = np.argwhere(y == label).reshape(-1)
        centroid = np.mean(np.array(X[i_labels, :]), axis=0)
        dist, ind = knn.kneighbors(centroid)
        if y[ind[0]] == label:
            i=ind[0]              
        # Add to new sets
        S.append(X[i, :])
        V.append(y[i])
        # Remove from new sets
        X = np.delete(X, i, axis=0)
        y = np.delete(y, i)

    S = np.array(S)
    V = np.array(V)
    knn.fit(S, V)
    indices_to_remove=list(S)
    
    while True:
        indices_to_remove=[]
        for i in range(S.shape[0]):
            # calculate distances between p = S[i,:] and others instances of S
            distances_S = knn._calculate_distance(S, S[i,:])
            distances_S = distances_S[distances_S != 0]
            min_dist = distances_S.min()
            distances_X=knn._calculate_distance(X, S[i,:])
            for dist_id, dist in enumerate(distances_X):
                if dist<min_dist:
                    if y[dist_id] != V[i]:
                        indices_to_remove.append(dist_in)      
        if len(indices_to_remove):
            indices_to_remove=list(set(indices_to_remove))
            # Update S and V
            S = np.vstack((S, X[indices_to_remove, :]))
            V = np.concatenate((V, y[indices_to_remove]))
            knn.fit(S, V)

            # Remove instance from X
            X = np.delete(X, indices_to_remove, axis=0)
            y = np.delete(y, indices_to_remove)
        else:
            break

    return S, V
