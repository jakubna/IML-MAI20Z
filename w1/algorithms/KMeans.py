import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
# only import sklearn for the metric of silhouette_score for reach the k optimal value.


class KMeans:
    def __init__(self, k: int, max_it=100, seed=-1, tol=1e-5):
        """
        :param k: Number of Clusters
        :param max_it: Maximum number of iterations if hasn't reached convergence yet.
        :param seed: Fixed seed to allow reproducibility.
        :param tol: Relative tolerance with regards difference in the cluster centers of two consecutive iterations to
        declare convergence.
        """
        if k < 1:
            raise ValueError('K must be a positive number')

        self.k = k
        self.max_it = max_it
        self.seed = seed
        self.tol = tol
        self.n_it = 0
        self.metric = 'euclidean'
        self.previous_centroids = None
        self.X = None
        self.centroids = None
        self.nearest = None

    def fit(self, x: np.ndarray):
        """
        Compute cluster centers and predict cluster index for each sample.
        :param x: 2D data array of size (rows, features).
        """
        if self.seed < 0:
            np.random.seed()
        else:
            np.random.seed(self.seed)

        self.X = x
        # Initialize centroids
        self._init_centroids()

        while True:
            self.n_it += 1
            distances = self._distances(x)
            labels, self.nearest, nearest_ids = self._get_nearest(x, distances)
            self.previous_centroids = self.centroids.copy()
            self._update_centroids()

            if self._check_convergence():
                break

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
    
    def optimal_k4db(self, x: np.ndarray, k_range=range(2, 10), plot=False):
        """
        Search the optimal number of clusters for the X dataset.
        :param x: 2D data array of size (rows, features).
        :param k_range: the range of different k that the function will evaluate.
        :param plot: this option pop up the Silhouette method plot.
        :return: the optimal value of cluster for X data base.
        """
        if k_range.start < 2:
            raise Exception('The minimum range starting number must be 2')
        ss_distances = []
        for k in k_range:
            km = KMeans(k, seed=self.seed)
            labels = km.fit_predict(x)
            ss_distances.append(silhouette_score(x, labels, metric='euclidean'))

        if plot:
            plt.plot(k_range, ss_distances, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Silhouette score')
            plt.title('The Silhouette Method For Optimal k')
            plt.show()

        return ss_distances.index(max(ss_distances)) + k_range.start

    def _init_centroids(self):
        """Initialize centroids"""
        init_centroids = np.random.choice(range(self.X.shape[0]), size=self.k, replace=False)
        self.centroids = self.X[init_centroids, :]

    def _calculate_sd(self):
        """
        Calculate the distance between old centroids and new ones (SD).
        :return: SD.
        """
        cost = 0
        for k in range(self.k):
            cost += \
                distance.cdist(np.array([self.centroids[k]]), np.array([self.previous_centroids[k]]),
                               metric=self.metric)[
                    0][0]
        return cost

    def _distances(self, x: np.ndarray):
        """
        Calculate distance from each point of the dataset to each cluster.
        :param x: 2D data array of size (rows, features).
        :return: Distance matrix of shape (K, number of points)
        """
        distances = np.zeros(shape=(self.k, x.shape[0]))

        for centroid_id, centroid in enumerate(self.centroids):
            for row_id, row in enumerate(x):
                distances[centroid_id, row_id] = self._calculate_distance(centroid, row)

        return distances

    def _calculate_distance(self, x: np.ndarray, y: np.ndarray):
        """
        Calculate distance between 2 elements using the metric depending on the algorithm ('euclidean' or 'cityblock').
        :param x: 1D vector with all x attributes.
        :param y: 1D vector with all y attributes.
        :return: Distance between both vectors using the specified metric.
        """
        return distance.cdist(np.array([x]), np.array([y]), metric=self.metric)[0][0]

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

    def _update_centroids(self):
        """Compute the new centroid for each cluster using method depending on algorithm."""
        for k in range(self.k):
            if len(self.nearest[k]) > 0:
                self.centroids[k, :] = np.mean(np.array(self.nearest[k]), axis=0)

    def _check_convergence(self):
        """Check the termination criterions"""
        if self.n_it >= self.max_it:
            return True
        elif self._calculate_sd() < self.tol:
            return True
        else:
            return False
