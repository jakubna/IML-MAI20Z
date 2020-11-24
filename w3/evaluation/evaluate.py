from sklearn.metrics import *
import numpy as np
import scipy.spatial


def evaluate_supervised_external(labels_true, labels_predicted):
    """
    this functions is to compare the predicted results with the real ones (supervised methods) in some different metrics
    :param labels_true: 1D data array of size (true labels).
    :param labels_predicted: 1D data array of size (predicted labels).
    """
    supervised_scores = dict(
        accuracy=accuracy_score(labels_true, labels_predicted),
        precision=precision_score(labels_true, labels_predicted),
        recall=recall_score(labels_true, labels_predicted),
        f1score=f1_score(labels_true, labels_predicted),
        adjusted_mutual_info_score=adjusted_mutual_info_score(labels_true, labels_predicted, 'arithmetic'),
        adjusted_rand_score=adjusted_rand_score(labels_true, labels_predicted),
        completeness_score=completeness_score(labels_true, labels_predicted),
        confusion_matrix=confusion_matrix(labels_true, labels_predicted),
        classification_report=classification_report(labels_true, labels_predicted),
        fowlkes_mallows_score=fowlkes_mallows_score(labels_true, labels_predicted),
        homogeneity_score=homogeneity_score(labels_true, labels_predicted),
        v_measure_score=v_measure_score(labels_true, labels_predicted))
    return supervised_scores


