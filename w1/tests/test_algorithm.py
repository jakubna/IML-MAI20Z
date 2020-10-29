from w1.dataPreprocessing.breast import preprocess as preprocess_breast
from w1.dataPreprocessing.cmc import preprocess as preprocess_cmc
from w1.dataPreprocessing.nursery import preprocess as preprocess_nursery
from w1.algorithms.KMeans import KMeans
from w1.algorithms.bisecting_kmeans import BisectingKMeans
from w1.algorithms.KMedians import KMedians
from w1.algorithms.FuzzyCMeans import FuzzyCMeans
from w1.algorithms.dbscan import dbscan_, find_eps
from w1.evaluation.evaluate import evaluate_supervised, evaluate_unsupervised, evaluate_soft_partitions

# preprocess the three datasets
X_bre, label_true_bre, df_bre = preprocess_breast()
X_cmc, label_true_cmc, df_cmc = preprocess_cmc()
X_nur, label_true_nur, df_nur = preprocess_nursery()

# apply algorithms
algorithm = BisectingKMeans(k=5, seed=2)
labels_bre_bisecting = algorithm.fit_predict(X_bre)
algorithm2 = KMeans(k=3, max_it=20)
labels_bre_kmeans = algorithm2.fit_predict(X_bre)
algorithm3 = KMedians(k=2, max_it=20)
labels_bre_kmedians = algorithm3.fit_predict(X_bre)
algorithm4 = FuzzyCMeans(k=2, max_it=20)
labels_bre_fuzzy = algorithm4.fit_predict(X_bre)
eps = find_eps(X_bre)
results, df = dbscan_(X_bre, df=df_bre, eps=0.2, min_s=3)

# apply evaluation
sup_kmeans = evaluate_supervised(label_true_bre, labels_bre_kmeans)
uns_kmeans = evaluate_unsupervised(X_bre, labels_bre_kmeans)
fuzzy_cmeans = evaluate_soft_partitions(X_bre, label_true_bre, labels_bre_fuzzy, algorithm4.centroids)
print(fuzzy_cmeans)
print(sup_kmeans)
print(uns_kmeans)
