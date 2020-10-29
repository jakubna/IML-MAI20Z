import numpy as np
from w1.algorithms.KMeans import KMeans


class FuzzyCMeans(KMeans):
    def __init__(self, m=2, epsilon=0.01, **kwargs):
        super().__init__(**kwargs)
        """
        :param m: the fuzziness index m € [1, ∞]
        :param epsilon: the termination criterion between [0, 1]
        """
        self.m = m
        self.epsilon = epsilon

    def predict(self, x: np.ndarray):
        """
        Override function
        :param x: 2D data array of size (rows, features).
        :return: Membership matrix,
                 Centroids parameters,
                 Cluster indexes assigned to each row of X.
        """
        crisp_labels = self._crisp_predict(x)
        return self.u, self.centroids, crisp_labels

    def _crisp_predict(self, x: np.ndarray):
        """
        Getting the clusters
        :param x: 2D data array of size (rows, features).
        :return: Cluster indexes assigned to each row of X.
        """
        cluster_labels = list()
        for i in range(x.shape[0]):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(self.u[:, i]))
            cluster_labels.append(idx)
        return cluster_labels

    def _init_centroids(self):
        """Override function: initialization of centroids and calling _init_u functon."""
        super()._init_centroids()
        self._init_u()

    def _init_u(self):
        """Initialization of membership matrix U (K, n) with random values such that each column SUM up to 1"""
        u = np.random.random(size=(self.k, self.X.shape[0]))
        self.u = u / u.sum(axis=0)

    def _update_centroids(self):
        """Shift of the centroids and calling _update_u function"""
        u_pow_m = self.u ** self.m
        numerator = np.dot(u_pow_m, self.X)
        denominator = u_pow_m.sum(axis=1, keepdims=True)
        self.centroids = numerator / denominator
        self._update_u()

    def _update_u(self):
        """Calculate the membership matrix U. Each column sum up to 1."""
        for k in range(self.k):
            for i in range(self.X.shape[0]):
                num_norm = np.linalg.norm(self.X[i, :] - self.centroids[k, :])
                u_ki = 0
                for j in range(self.k):
                    den_norm = np.linalg.norm(self.X[i, :] - self.centroids[j, :])
                    u_ki += (num_norm / den_norm) ** (2 / (self.m - 1))
                u_ki **= -1
                self.u[k, i] = u_ki
        self.u = self.u / self.u.sum(axis=0)

    def _check_convergence(self):
        """Check the termination criteria"""
        if self.n_it >= self.max_it:
            return True
        elif np.linalg.norm(self.centroids - self.previous_centroids, ord=1) < self.epsilon:
            return True
        else:
            return False
