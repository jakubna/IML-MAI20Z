import numpy as np
from w1.algorithms.KMeans import KMeans
from scipy.spatial import distance


def sse(cluster):
    """
    calculate the error of the cluster.
    :param cluster:  2D data array of size (rows, features).
    :return: sum of errors
    """
    centroid = np.mean(cluster, 0)
    errors = np.linalg.norm(cluster - centroid, ord=2, axis=1)
    return np.sum(errors)


class BisectingKMeans:
    def __init__(self, k: int, max_it=100, seed=1, tol=1e-5):
        """
        :param k: Number of Clusters
        :param max_it: Maximum number of iterations if hasn't reached convergence yet (for KMeans fit)
        :param seed: Fixed seed to allow reproducibility (for KMeans fit)
        :param tol: Relative tolerance with regards difference in the cluster centers of two consecutive iterations to
        declare convergence (for KMeans fit).
        """
        if k < 1:
            raise ValueError('K must be a positive number')

        self.k = k
        self.metric = 'euclidean'
        self.max_it = max_it
        self.seed = seed
        self.tol = tol
        self.centroids = None
        self.nearest = None

    def fit(self, x: np.ndarray):
        """
        Compute cluster centers and predict cluster index for each sample.
        :param x: 2D data array of size (rows, features).
        """
        clusters = [x]
        while len(clusters) < self.k:
            # calculate the cluster with the biggest error
            max_sse_i = np.argmax([sse(c) for c in clusters])
            # select the cluster and take it off from the clusters list
            cluster = clusters.pop(max_sse_i)
            # split in 2 clusters using k_means
            kmeans = KMeans(k=2, max_it=self.max_it, seed=self.seed, tol=self.tol)
            print(cluster)
            kmeans.fit(cluster)
            two_labels = kmeans.predict(cluster)
            # use the labels to split the data according to clusters
            two_clusters = []
            cluster_x = []
            for act in range(0, 2):
                # select the index of all points in the same cluster
                index = np.where(two_labels == act)[0]
                cluster_x = cluster[index, :]
            two_clusters.append(cluster_x)
            # append the clusters list
            clusters.extend(two_clusters)

        centroids = []
        for c in clusters:
            centroids.append(np.mean(c, 0))
            self.centroids = centroids

    def _get_nearest(self, x: np.ndarray, distances: np.ndarray):
        """
        Compute the distance for each dataset instance to each centroid.
        :param x: 2D data array of size (rows, features).
        :param distances: 2D vector of distances between centroids and points.
        :return: Cluster indexes assigned to each observation (labels for each point).
                 List of nearest observations for each cluster.
                 List of nearest observations index for each cluster.
        """
        clusters = []
        nearest = [[] for _ in range(self.k)]
        nearest_id = [[] for _ in range(self.k)]

        for row_id, row in enumerate(x):
            cluster_id = int(np.argmin(distances[:, row_id]))

            clusters.append(cluster_id)
            nearest[cluster_id].append(row)
            nearest_id[cluster_id].append(row_id)

        return clusters, nearest, nearest_id

    def predict(self, x: np.ndarray):
        """
       Assign labels to a list of observations.
       :param x: 2D data array of size (rows, features).
       :return: Cluster indexes assigned to each row of X.
       """
        if self.centroids is None:
            raise Exception('Fit the model with some data before running a prediction')

        distances = self._distances(x)
        labels, self.nearest, nearest_ids = self._get_nearest(x, distances)

        return labels

    def fit_predict(self, x: np.ndarray):
        """
        Fit the model with data and return assigned labels.
        :param x: 2D data array of size (rows, features).
        :return: Cluster indexes assigned to each row of X.
        """
        self.fit(x)
        return self.predict(x)

    def _distances(self, x: np.ndarray):
        """
        Calculate distance from each point of the dataset to each cluster.
        :param x: 2D data array of size (rows, features).
        :return: Distance matrix of shape (K, number of points)
        """
        distances = np.zeros(shape=(self.k, x.shape[0]))

        for centroid_id, centroid in enumerate(self.centroids):
            for row_id, row in enumerate(x):
                distances[centroid_id, row_id] = distance.cdist(np.array([centroid]), np.array([row]),
                                                                metric=self.metric)[0][0]

        return distances
