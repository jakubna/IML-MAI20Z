import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def pca_sklearn(X, datasetname):

""" function to compute PCA using sklearn algorithm
    X: 2D data array of size (rows, features)
    datasetname: string (name of data set to set plot title)
    return: engeivectors, engeinvalues , reduced data set
"""

  pca = PCA(n_components=len(X[0]))
  X_reduced = pca.fit_transform(X)
  
  #compute the engeinvalues and engevectors and reduced data to the number of components that have engeinvalue bigger than 1 (more representative, has bigger variance)
  bigest_eigenvalues=[i for i in pca.singular_values_ if i>1 ]
  X_pca= X_reduced[:,:len(bigest_eigenvalues)]
  eigenvectors = pca.components_[:len(bigest_eigenvalues),:]
  
  #plot the components and engeinvalues to understand the number of optimal components
  x = np.arange(len(pca.singular_values_))
  labels = [ str(i+1)+ 'º Component' for i in list(x)]
  plt.bar(x,pca.singular_values_)
  plt.title('PCA - ' + dataset + ' data set')
  plt.ylabel('Eingevalue')
  plt.xticks(x, labels)
  plt.show()

  return bigest_eigenvalues, eigenvectors, X_pca
