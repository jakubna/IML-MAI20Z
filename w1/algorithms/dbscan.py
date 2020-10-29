import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN 
from sklearn.neighbors import NearestNeighbors


def find_eps(x):
    """ this function aims to plot the epsilon by calculating the distance to the nearest two points and index.
      the optimal value will be found at the point of maximum curvature  
      x: normalized input variables
    """
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(x)
    distances, indices = nbrs.kneighbors(x)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)


def dbscan_(x, df, eps):
    """ this functions aims to clusterize the dataset using sklearn DBSCAN using different metrics and algorithms
        and returns the original dataframe with labels of each metric-algorithm 
                     and a dataframe with results (columns = metric, algorithm, number of clusters, number of estimated
                     noise points.
        x: normalized input variables
        df: original dataframe with variables and true labels
        eps: optimal epsilon (obtained by the graph using find_eps function
        min_s: minimal number of samples in one cluster """
        
    metrics_ = ['euclidean', 'cosine', 'l1', 'l2', 'manhattan']
    algo = ['auto', 'ball_tree', 'kd_tree', 'brute']
    metrics_list = []
    algorithms_list = []
    clusters_list = []
    noise_list = []
    for metric in metrics_:  # iterating to combine every metric with all algorithms
        for method in algo:
            ma = metric+method
            if ma != 'cosineball_tree' and ma != 'cosinekd_tree':  # cosine can't deal wth ball_tree and kd_tree algrtms
                db = DBSCAN(eps=eps, min_samples=int(np.log(len(X))), metric=metric, algorithm=method).fit(x)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_
                df[metric+'-'+method] = labels
                # labels_true = Y
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # taking off the noise points
                n_noise_ = list(labels).count(-1)
                metrics_list.append(metric)
                algorithms_list.append(method)
                clusters_list.append(n_clusters_)
                noise_list.append(n_noise_)

    df_results = pd.DataFrame(columns=['Metric', 'Algorithm', 'Estimated Clusters', 'Estimated noise points'])
    df_results['Metric'] = metrics_list
    df_results['Algorithm'] = algorithms_list
    df_results['Estimated Clusters'] = clusters_list
    df_results['Estimated noise points'] = noise_list

    return df_results, df
