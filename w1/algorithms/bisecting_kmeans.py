import numpy as np
from algorithms.KMeans import KMeans

class Bisecting_KMeans:
    def __init__(self, k: int, max_it=100, seed=1, tol=1e-5):
        """
        :param k: Number of Clusters 
        :param max_it: Maximum number of iterations if hasn't reached convergence yet (for KMeans fit)
        :param seed: Fixed seed to allow reproducibility (for KMeans fit)
        :param tol: Relative tolerance with regards difference in the cluster centers of two consecutive iterations to declare convergence (for KMeans fit)
        """
        if k < 1:
            raise ValueError('K must be a positive number')

        self.k = k
        self.metric = 'euclidean'
    
    def fit(self, X: np.ndarray):
        """
        Compute cluster centers and predict cluster index for each sample.
        :param X: 2D data array of size (rows, features).
        """
        clusters = [X]
        while len(clusters) < k:
             #calculate the cluster with the biggest error
             max_sse_i = np.argmax([SSE(c) for c in clusters])
             #select the cluster and take it off from the clusters list
             cluster = clusters.pop(max_sse_i)
             #split in 2 clusters using k_means
             kmeans = k_means(k=2, max_it, seed, tol)
             kmeans.fit(cluster)
             two_labels = kmeans.predict(cluster)
             #use the labels to split the data according to clusters
             two_clusters=[]
             for act in range(0, 2):
                 # select the index of all points in the same cluster
                 indX = np.where(two_labels == act)[0]
                 cluster_x = cluster[indX, :]
                two_clusters.append(cluster_x)
             #append the clusters list
             clusters.extend(two_clusters)
        
        centroids = []
        for c in clusters:
            centroids.append(np.mean(c, 0)
        self.centroids = centroids
                             
                             
    def predict(self, X: np.ndarray):
         """
        Assign labels to a list of observations.
        :param X: 2D data array of size (rows, features).
        :return: Cluster indexes assigned to each row of X.
        """
        if self.centroids is None:
            raise Exception('Fit the model with some data before running a prediction')

        distances = self._distances(X)
        labels, self.nearest, nearest_ids = self._get_nearest(X, distances)

        return labels

    def fit_predict(self, X: np.ndarray):
        """
        Fit the model with data and return assigned labels.
        :param X: 2D data array of size (rows, features).
        :return: Cluster indexes assigned to each row of X.
        """
        self.fit(X)
        return self.predict(X)            


    def SSE(cluster):
        """ 
        calculate the error of the cluster.
        :param cluster:  2D data array of size (rows, features).
        :return: sum of errors 
        """
        centroid = np.mean(cluster, 0)
        errors = np.linalg.norm(cluster-centroid, ord=2, axis=1)
        return np.sum(errors)
                             
    def _distances(self, X: np.ndarray):
        """
        Calculate distance from each point of the dataset to each cluster.
        :param X: 2D data array of size (rows, features).
        :return: Distance matrix of shape (K, number of points)
        """
        distances = np.zeros(shape=(self.k, X.shape[0]))

        for centroid_id, centroid in enumerate(self.centroids):
            for row_id, row in enumerate(X):
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

    def _get_nearest(self, X: np.ndarray, distances: np.ndarray):
        """
        Compute the distance for each dataset instance to each centroid.
        :param X: 2D data array of size (rows, features).
        :param distances: 2D vector of distances between centroids and points.
        :return: Cluster indexes assigned to each observation (labels for each point).
                 List of nearest observations for each cluster.
                 List of nearest observations index for each cluster.
        """
        clusters = []
        nearest = [[] for _ in range(self.k)]
        nearest_id = [[] for _ in range(self.k)]

        for row_id, row in enumerate(X):
            cluster_id = int(np.argmin(distances[:, row_id]))

            clusters.append(cluster_id)
            nearest[cluster_id].append(row)
            nearest_id[cluster_id].append(row_id)

        return clusters, nearest, nearest_id
            

