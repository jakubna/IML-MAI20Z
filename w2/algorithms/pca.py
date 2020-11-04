from numpy import linalg as LA
import numpy as np

class PCA:
    
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean = None
        self.cov_matrix = None
        self.eig_val = None
        self.components = None

    def fit(self, X: np.ndarray):
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
        self.eig_val = eig_val[:self.n_components]
        self.components = eig_vect[:self.n_components, :]
        print('K first eigenvalues:\n', self.eig_val)
        print('K eigenvectors:\n', self.components)

    def transform(self, X: np.ndarray):
        X_transformed = np.dot(self.components, (X - self.mean).T).T
        return X_transformed

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)
