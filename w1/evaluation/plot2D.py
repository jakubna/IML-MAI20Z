import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot2D (X , y_predict, title:str):
  """ the function aims to reduce the features to two dimensions using PCA method and plot the clusters
      X: input dataset
      y_predict: results labels
      title: graphic title (example: adults dataset using Kmeans)"""
  pca = PCA(n_components=2)
  principalComponents = pca.fit_transform(X)
  finalDf = pd.DataFrame(data = principalComponents
               , columns = ['principal component 1', 'principal component 2'])
  fig = plt.figure(figsize = (8,8))
  ax = fig.add_subplot(1,1,1) 
  ax.set_xlabel('Principal Component 1', fontsize = 15)
  ax.set_ylabel('Principal Component 2', fontsize = 15)
  ax.set_title('2 component PCA' + title, fontsize = 20)
  targets = y_predict.unique()
  colors = ['r', 'g', 'k', 'y','c','m']
  for target, color in zip(targets,colors):
      indicesToKeep = y_predict == target
      ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                  , finalDf.loc[indicesToKeep, 'principal component 2']
                 , c = color
                 , s = 50)
  ax.legend(targets)
  ax.grid()
