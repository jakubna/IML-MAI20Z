from typing import List
from w1.evaluation.evaluate import *
from w1.algorithms.FuzzyCMeans import FuzzyCMeans


def optimize(x: np.ndarray, y: np.ndarray, algorithm, metric: str, k_values: List[int], goal: str, seed: int):
    """
    Optimize K value for the same data, algorithm and metric.
    :param x: 2D data matrix of size (#observations, #features).
    :param y: 1D data matrix of size (#true_labels).
    :param algorithm: Algorithm class to instantiate.
    :param seed: seed algorithm
    :param metric: Metric function used to evaluate - possible: adjusted_mutual_info_score,adjusted_rand_score,
                                                                    completeness_score, fowlkes_mallows_score,
                                                                    homogeneity_score,v_measure_score,
                                                                    calinski_harabasz_score, davies_bouldin_score,
                                                                    silhouette_score,normalized_partition_coefficient,
                                                                    partition_entropy, xie_beni
    :param k_values: List of `K` values to test.
    :param goal: `max` or `min` the metric (maximaze or minimize the metric)
    :return: list containing k values and scores
    """
    results = []
    plot_res = []
    score = 0
    assert goal in ['max', 'min']
    for k in k_values:  # iterate for every k in list
        model = algorithm(k=k, seed=seed)  # algorithm params
        model.fit(x)
        if algorithm == FuzzyCMeans:
            y_pred = model.predict(x)[2]
        else:
            y_pred = model.predict(x)
        if metric in ['adjusted_mutual_info_score', 'adjusted_rand_score', 'completeness_score',
                      'fowlkes_mallows_score', 'homogeneity_score', 'v_measure_score']:
            scores = evaluate_supervised_external(y, y_pred)
            score = scores[metric]
        if metric in ['calinski_harabasz_score', 'davies_bouldin_score', 'silhouette_score']:
            scores = evaluate_unsupervised_internal(x, y_pred)
            score = scores[metric]
        if metric in ['normalized_partition_coefficient', 'partition_entropy', 'xie_beni']:
            scores = evaluate_soft_partitions_internal(x, y_pred[0], y_pred[1])
            score = scores[metric]
        results.append({'k': k, 'score': score})
        plot_res.append(score)

    return dict(optimal=(sorted(results, key=lambda z: z['score'], reverse=goal == 'max')), plot=plot_res)
