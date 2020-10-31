from w1.evaluation.optimize import *
from w1.algorithms.KMedians import KMedians
from w1.algorithms.KMeans import KMeans
from w1.algorithms.bisecting_kmeans import BisectingKMeans
from w1.dataPreprocessing.breast import preprocess as preprocess_breast
from w1.dataPreprocessing.nursery import preprocess as preprocess_nursery
from w1.dataPreprocessing.cmc import preprocess as preprocess_cmc
from w1.algorithms.FuzzyCMeans import FuzzyCMeans
processed = preprocess_cmc()
print('CMC')
teste=optimize(x = processed['db'], y=processed['label_true'] ,algorithm=KMeans,  metric='silhouette_score',
             k_values = [2,3,4,5,6,7,8,9,10], goal ='max')
print('KMeans',teste[0])
teste=optimize(x = processed['db'], y=processed['label_true'] ,algorithm=KMedians,  metric='silhouette_score',
             k_values = [2,3,4,5,6,7,8,9,10], goal ='max')
print('KMedians',teste[0])
teste=optimize(x = processed['db'], y=processed['label_true'] ,algorithm=BisectingKMeans,  metric='silhouette_score',
             k_values = [2,3,4,5,6,7,8,9,10], goal ='max')
print('BisectingKMeans',teste[0])
teste=optimize(x = processed['db'], y=processed['label_true'] ,algorithm=FuzzyCMeans,  metric='silhouette_score',
             k_values = [2,3,4,5,6,7,8,9,10], goal ='max')
print('Fuzzy',teste[0])
