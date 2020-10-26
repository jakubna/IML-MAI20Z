import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN 
from sklearn.neighbors import NearestNeighbors

def find_eps(X):
  neigh = NearestNeighbors(n_neighbors=2)
  nbrs = neigh.fit(X)
  distances, indices = nbrs.kneighbors(X)
  distances = np.sort(distances, axis=0)
  distances = distances[:,1]
  plt.plot(distances)
  
def dbscan_(X, df, eps, min_s):
  metrics_=['euclidean','cosine','l1','l2','manhattan']
  algo =['auto', 'ball_tree', 'kd_tree', 'brute']
  metrics_list=[]
  algorithms_list=[]
  clusters_list=[]
  noise_list=[]
  for metric in metrics_:
      for method in algo:
          ma=metric+method
          if ma != 'cosineball_tree' and  ma != 'cosinekd_tree':
              db = DBSCAN(eps=eps,min_samples=min_s,metric=metric, algorithm=method).fit(X)
              core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
              core_samples_mask[db.core_sample_indices_] = True
              labels = db.labels_
              df[metric+'-'+method]=labels
              labels_true=Y
              n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
              n_noise_ = list(labels).count(-1)
              metrics_list.append(metric)
              algorithms_list.append(method)
              clusters_list.append(n_clusters_)
              noise_list.append(n_noise_)

  df_results = pd.DataFrame(columns=['Metric','Algorithm','Estimated Clusters','Estimated noise points'])
  df_results['Metric']=metrics_list
  df_results['Algorithm']=algorithms_list
  df_results['Estimated Clusters']=clusters_list
  df_results['Estimated noise points']=noise_list
  return df_results, df

