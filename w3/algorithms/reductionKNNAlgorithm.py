import numpy as np

from algorithms.kNNAlgorithm import kNNAlgorithm
from algorithms.reduction.drop3 import drop3_reduction
from algorithms.reduction.enn import enn_reduction
from algorithms.reduction.menn import menn_reduction
from algorithms.reduction.fcnn import fcnn_reduction

reduction_methods = ['drop3', 'enn', 'fcnn', 'menn']

def reduction_KNNAlogrithm(config: dict, X: np.ndarray, y: np.ndarray, reduction_method: str):
    if reduction_method not in reduction_methods:
        raise ValueError('Unknown reduction method')
    X = X.copy()
    y = y.copy()
    alg = kNNAlgorithm(**config)

    if reduction_method == 'drop3':
        return drop3_reduction(alg, X, y)
    elif reduction_method == 'enn':
        return enn_reduction(alg, X, y)
    elif reduction_method == 'fcnn':
        return menn_reduction(alg, X, y)
    elif reduction_method == 'menn':
        return fcnn_reduction(alg, X, y)