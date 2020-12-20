import logging
from typing import Tuple, List, Set
import numpy as np
import pandas as pd
from algorithms.kNNAlgorithm import kNNAlgorithm
from copy import deepcopy


def drop3_reduction(knn: kNNAlgorithm, X: np.ndarray, y: np.ndarray):
    knn.fit(X, y)
    remove_intances = []
    for i in range(X.shape[0]):
        dist, kn = knn.kneighbors([X[i, :]], return_distance=True)
        kn = kn.tolist()
        kn = kn[0]
        y_kn = [y[j] for j in kn]
        if len(set(y_kn)) > 1:
            remove_intances.append(i)
        else:
            if list(set(y_kn))[0] != y[i]:
                remove_intances.append(i)

    if len(remove_intances):
        # Remove instances from original data
        X = np.delete(X, remove_intances, axis=0)
        y = np.delete(y, remove_intances)

    S = list(range(X.shape[0]))
    associates: List[Set[int]] = [set() for _ in range(X.shape[0])]
    # neighbours = [[] for _ in range(S.shape[0])]
    if knn.metric== 'cityblock':
        metric_= 'minkowski'
    else:
        metric_=knn.metric
    knn_1 = kNNAlgorithm(n_neighbors=knn.n_neighbors + 1, policy=knn.policy,  metric=metric_)
    knn_1.fit(X, y)

    for p_idx in range(X.shape[0]):
        # Find the k + 1 nearest neighbors of p in S.
        associates[p_idx] = set(knn_1.kneighbors([X[p_idx, :]])[0])
        logging.debug(f'Instance {p_idx} neighbours -> {associates[p_idx]}')

        # Add p to each of its neighbors’ lists of associates.
        for n_idx in associates[p_idx]:
            associates[n_idx].add(p_idx)
            logging.debug(f'- Associates of {n_idx} -> {associates[n_idx]}')

    # Order X from the element that has the furthest to the nearest nearest enemy
    # (nearest neighbor that is of a different class), to do that:

    min_dist = []
    instances = []
    classes=[]
    for p in range(X.shape[0]):
        enemies = []
        yp = y[p]
        for i in range(X.shape[0]):
            if y[i] != yp:
                enemies.append(i)
        distances = knn._calculate_distance([X[p, :]], X[enemies, :])
        min_dist.append(min(distances[0]))
        instances.append(X[p, :])
        classes.append(y[p])
    df = pd.DataFrame()
    df['x'] = instances
    df['dist'] = min_dist
    df['y']=classes
    df = df.sort_values(['dist'], ascending=False)
    X = np.array(list(df['x']))
    y=np.array(list(df['y']))


    p_idx = 0
    p_idx_original = 0
    while p_idx < len(S):
        # Num. of associates of p classified correctly with p as a neighbour.
        knn.fit(X[S], y[S])
        try:
            d_with = sum(map(lambda x: y[x] == knn.kneighbors(X[x]), associates[p_idx]))
        except:
            p_idx += 1
            continue
        # Num. of associates of p classified correctly without p as a neighbour.
        # knn.fit(np.delete(S, p_idx, axis=0), np.delete(V, p_idx))
        S_ = S.copy()
        del S_[p_idx]
        knn.fit(X[S_], y[S_])
        try:
            d_without = sum(map(lambda x: y[x] == knn.kneighbors(X[x]), associates[p_idx]))
        except:
            p_idx += 1
            continue

        logging.debug(f'For instance {p_idx}: with={d_with} and without={d_without}')

        if d_without >= d_with:
            S = S_
            for a_idx in associates[p_idx_original] - {p_idx_original}:
                # Remove p from a’s list of nearest neighbors.
                associates[a_idx] -= {p_idx_original}

                # Find a new nearest neighbor for A.
                a_nn = knn.kneighbors(X[a_idx, :])

                # Add A to its new neighbor’s list of associates
                associates[a_nn[0]].add(a_idx)

        else:
            p_idx += 1
        p_idx_original += 1

    return X[S], y[S]
