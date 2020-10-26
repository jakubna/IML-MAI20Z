import numpy as np
from sklearn.metrics import *
from typing import Type, List
from evaluation.evaluate import *

    """
    Optimize K value for the same data, algorithm and metric.
    :param X: 2D data matrix of size (#observations, #features).
    :param X: 1D data matrix of size (#true_labels).
    :param algorithm: Algorithm class to instantiate.
    :param algorithm_params: Extra parameters for the algorithm class.
    :param metric: Metric function used to evaluate - possible: adjusted_mutual_info_score,adjusted_rand_score,
                                                                completeness_score, fowlkes_mallows_score,
                                                                homogeneity_score,v_measure_score,calinski_harabasz_score,
                                                                davies_bouldin_score,silhouette_score,normalized_partition_coefficient,
                                                                partition_entropy, xie_beni
    :param k_values: List of `K` values to test.
    :param goal: `max` or `min` the metric.
    :return: list containg k values and scores
    """
def optimize(X: np.ndarray,y:np.ndarray, algorithm: Type[KMeans], algorithm_params: dict, metric: str, k_values: List[int], goal: str):
    results = []
    assert goal in ['max', 'min']
    for k in k_values:
        model = algorithm(k=k,algorithm_params)
        model(X).fit
        y_pred = model.predict(X)
        if metric in ['adjusted_mutual_info_score','adjusted_rand_score','completeness_score', 'fowlkes_mallows_score','homogeneity_score','v_measure_score']:
            scores= evaluate_supervised(y,y_pred)
            score=scores[metric]
        if metric in ['calinski_harabasz_score','davies_bouldin_score','silhouette_score']:
            scores=evaluate_unsupervised(X,y_pred)
            score=scores[metric]
        if metric in ['normalized_partition_coefficient','partition_entropy', 'xie_beni']:
            scores=evaluate_soft_partitions(X,y,y_pred,model.centroids)
            score=scores[metric]
        results.append({'k':k,'score':score})
    return sorted(results, key=lambda x:x['score'], reverse=goal == 'max')
        
    
