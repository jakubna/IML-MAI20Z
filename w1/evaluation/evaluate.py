from sklearn.metrics import *
import numpy as np
import scipy.spatial


def get_metrics_from_mat(contingency_matrix):
    """
    Function to compute precision, recall, f1 score and accuracy
    :param contingency_matrix: 2D data array of size.
    :return: precision (list)
        recall (list)
        f1 score (list)
        accuracy
    """
    precision = []
    recall = []
    f1score = []
    for i in range(len(contingency_matrix)):
        if i < contingency_matrix.shape[1]:
            p = (contingency_matrix[i][i]/sum(contingency_matrix[:, i]))
            r = contingency_matrix[i][i]/sum(contingency_matrix[i, :])
            precision.append(p)
            recall.append(r)
            if r + p > 0:
                f1score.append(2*((p*r)/(p+r)))
            else:
                f1score.append(0)
    accuracy = np.trace(contingency_matrix) / np.sum(contingency_matrix)
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1score=f1score)


def evaluate_supervised_external(labels_true, labels_predicted):
    """
    this functions is to compare the predicted results with the real ones (supervised methods) in some different metrics
    :param labels_true: 1D data array of size (true labels).
    :param labels_predicted: 1D data array of size (predicted labels).
    """
    contingency_matrix = cluster.contingency_matrix(labels_true, labels_predicted)
    metrics_from_contingency_matrix = get_metrics_from_mat(contingency_matrix)
    supervised_scores = dict(
        adjusted_mutual_info_score=adjusted_mutual_info_score(labels_true, labels_predicted, 'arithmetic'),
        adjusted_rand_score=adjusted_rand_score(labels_true, labels_predicted),
        completeness_score=completeness_score(labels_true, labels_predicted),
        contingency_matrix=contingency_matrix,
        fowlkes_mallows_score=fowlkes_mallows_score(labels_true, labels_predicted),
        homogeneity_score=homogeneity_score(labels_true, labels_predicted),
        v_measure_score=v_measure_score(labels_true, labels_predicted), **metrics_from_contingency_matrix)
    return supervised_scores


def evaluate_unsupervised_internal(x: np.ndarray, labels_predicted):
    """
    this functions is to evaluate the predicted results (unsupervised methods) by three metrics
    :param x : 2D data array of size (rows, features).
    :param labels_predicted: 1D data array of size (predicted labels).
    """
    unsupervised_scores = dict(calinski_harabasz_score=calinski_harabasz_score(x, labels_predicted),
                               davies_bouldin_score=davies_bouldin_score(x, labels_predicted),
                               silhouette_score=silhouette_score(x, labels_predicted))
    return unsupervised_scores


def evaluate_soft_partitions_internal(x: np.ndarray, membership_matrix, centroids):
    """
    this functions is to evaluate the predicted results by soft partitions methods
    :param x : 2D data array of size (rows, features).
    :param membership_matrix:  membership matrix, with shape (# clusters, # rows)
    :param centroids: 2d data array of size (k (number of clusters), features)
    """
    # calculations for xie_beni
    centroids_dist = scipy.spatial.distance.cdist(centroids, centroids) ** 2
    centroids_dist[centroids_dist == 0.0] = np.inf
    scores_soft_partitions = dict(normalized_partition_coefficient=(np.sum(membership_matrix ** 2) /
                                                                    membership_matrix.shape[1] - 1 /
                                                                    membership_matrix.shape[0]) / (
                                                                               1 - 1 / membership_matrix.shape[0]),
                                  partition_entropy=-np.sum(
                                      membership_matrix * np.log(membership_matrix) / membership_matrix.shape[1]),
                                  xie_beni=np.sum((membership_matrix ** 2).T * (
                                              scipy.spatial.distance.cdist(x, centroids) ** 2)) / (
                                                       x.shape[0] * np.min(centroids_dist)))
    return scores_soft_partitions

