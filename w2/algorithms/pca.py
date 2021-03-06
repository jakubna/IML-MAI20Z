from numpy import linalg as LA
import numpy as np
from typing import Union

class PCA:

    def __init__(self, n_components: Union[int, float, None] = None):
        """
        :param n_components: Number of components to keep. if n_components is not set all components are kept
        """
        self.n_components = n_components
        self.mean = None
        self.cov_matrix = None
        self.eig_val = None
        self.components = None
        self.all_eig_val = None
        self.all_eig_vect = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.k = None

    def fit(self, X: np.ndarray):
        """
        Fit the model with X.
        :param X: 2D data array of size (rows, features).
        """
        # Compute the d-dimensional mean vector
        self.mean = np.mean(X, axis=0)
        # Center columns by subtracting column means
        X_centered = (X - self.mean)
        # Compute the covariance matrix of the whole data set
        self.cov_matrix = np.cov(X_centered.T)
        print('Covariance Matrix:\n', self.cov_matrix)
        # Calculate eigenvectors and their corresponding eigenvalues of the covariance matrix
        eig_val, eig_vect = LA.eig(self.cov_matrix)
        print('Eigenvalues:\n', eig_val)
        print('Eigenvectors:\n', eig_vect.T)
        # Sort the eigenvectors by decreasing eigenvalues
        eig_map = list(zip(eig_val, eig_vect.T))
        print(eig_map)
        eig_map.sort(key=lambda x: x[0], reverse=True)
        eig_val, eig_vect = zip(*eig_map)
        eig_val, eig_vect = np.array(eig_val), np.array(eig_vect)
        self.all_eig_val = eig_val
        self.all_eig_vect = eig_vect
        # set explained variance and explained variance ratio
        self.explained_variance = eig_val
        self.explained_variance_ratio = self.explained_variance / self.explained_variance.sum()

        # setting number of components
        if type(self.n_components) == int:
            k = self.n_components
        elif type(self.n_components) == float:
            k = np.searchsorted(self.explained_variance_ratio.cumsum(), self.n_components) + 1
        else:
            k = X.shape[1]

        self.explained_variance = self.explained_variance[:k]
        self.explained_variance_ratio = self.explained_variance_ratio[:k]

        self.eig_val = eig_val[:k]
        self.components = eig_vect[:k, :]
        self.k = k
        print('K first eigenvalues:\n', self.eig_val)
        print('K eigenvectors:\n', self.components)

    def transform(self, X: np.ndarray):
        """
        Apply dimensionality reduction to X.
        :param X: 2D data array of size (rows, features).
        :return: Transformed X.
        """
        X_transformed = np.dot(self.components, (X - self.mean).T).T
        return X_transformed

    def fit_transform(self, X: np.ndarray):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        :param X: 2D data array of size (rows, features).
        """
        self.fit(X)
        return self.transform(X)
