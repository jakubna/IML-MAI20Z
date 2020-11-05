import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
import numpy as np

def pca_sklearn(X, datasetname):

""" function to compute PCA using sklearn algorithm
    X: 2D data array of size (rows, features)
    datasetname: string (name of data set to set plot title)
    return: engeivectors, engeinvalues , reduced data set
"""

  ipca = IncrementalPCA(n_components=len(X[0]))
  X_reduced_ipca = ipca.fit_transform(X)
  
  #compute the engeinvalues and engevectors and reduced data to the number of components that have engeinvalue bigger than 1 (more representative, has bigger variance)
  bigest_eigenvalues_ipca=[i for i in ipca.singular_values_ if i>1 ]
  X_ipca= X_reduced[:,:len(bigest_eigenvalues_ipca)]
  eigenvectors_ipca = ipca.components_[:len(bigest_eigenvalues_ipca),:]
  
  #plot the components and engeinvalues to understand the number of optimal components
  xipca = np.arange(len(ipca.singular_values_))
  labels_ipca = [ str(i+1)+ 'º Component' for i in list(xipca)]
  plt.title('Incremental PCA - '+ datasetname + ' data set')
  plt.ylabel('Eingevalue')
  plt.bar(xipca,ipca.singular_values_)
  plt.xticks(xipca, labels_ipca)
  plt.show()

  return bigest_eigenvalues_ipca, eigenvectors_ipca, X_ipca
