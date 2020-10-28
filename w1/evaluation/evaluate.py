from sklearn.metrics import *
import numpy as np
import scipy.spatial

def evaluate_supervised(labels_true, labels_predicted):
    """ 
    this functions is to compare the predicted results with the real ones (supervised methods) in some different metrics
    :param labels_true: 1D data array of size (true labels).
    :param labels_predicted: 1D data array of size (predicted labels).
    """    
    contingency_matrix = cluster.contingency_matrix(labels_true, labels_predicted)
    metrics_from_contingency_matrix = get_metrics_from_mat(contingency_matrix)
    supervised_scores = dict(adjusted_mutual_info_score=adjusted_mutual_info_score(labels_true, labels_predicted, 'arithmetic'),
               adjusted_rand_score=adjusted_rand_score(labels_true, labels_predicted),
               completeness_score=completeness_score(labels_true, labels_predicted),
               contingency_matrix=contingency_matrix,
               fowlkes_mallows_score=fowlkes_mallows_score(labels_true, labels_predicted),
               homogeneity_score=homogeneity_score(labels_true, labels_predicted),
               v_measure_score=v_measure_score(labels_true, labels_predicted), **metrics_from_contingency_matrix)
    return supervised_scores

def evaluate_unsupervised(X, labels_predicted):
     """ 
     this functions is to evaluate the predicted results (unsupervised methods) by three metrics
     :param X : 2D data array of size (rows, features).
     :param labels_predicted: 1D data array of size (predicted labels).
     """    
    unsupervised_scores = dict(calinski_harabasz_score=calinski_harabasz_score(X, labels_predicted),
                   davies_bouldin_score=davies_bouldin_score(X, labels_predicted),
                   silhouette_score=silhouette_score(X, labels_predicted))
    return unsupervised_scores


def evaluate_soft_partitions(X, labels_true, lables_predicted, centroids):
    """ 
    this functions is toevaluate the predicted results by soft partitions methods
    :param X : 2D data array of size (rows, features).
    :param labels_true: 1D data array of size (true labels).
    :param labels_predicted: 1D data array of size (predicted labels).
    :param centroids: 2d data array of size (k (number of clusters), features)
    """  
            
    contingency_matrix = cluster.contingency_matrix(labels_true, labels_predicted)
    #calculations for xie_beni
    centroids_dist = scipy.spatial.distance.cdist(centroids, centroids)**2
    centroids_dist[centroids_dist == 0.0] = np.inf  
    scores_soft_partitions = dict(normalized_partition_coefficient=(np.sum(contingency_matrix ** 2) / contingency_matrix.shape[1] - 1 / contingency_matrix.shape[0]) / (1 - 1 / contingency_matrix.shape[0]),
               partition_entropy=-np.sum(contingency_matrix * np.log(contingency_matrix) / contingency_matrix.shape[1]),
               xie_beni=np.sum((contingency_matrix**2).T*(scipy.spatial.distance.cdist(X, centroids)**2))/(X.shape[0]*np.min(centroids_dist)) )
    return scores_soft_partitions
