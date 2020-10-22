import pandas as pd
import numpy as np
import random as rd
from scipy.spatial import distance

class KMeans:
    def __init__(self, k:int, max_it=100, seed=1, tol=1e-4):
        """
        :param k: Number of Clusters
        :param max_it: Maximum number of iterations if hasn't reached convergence yet.
        :param seed: Fixed seed to allow reproducibility.
        :param tol: Relative tolerance with regards difference in the cluster centers of two consecutive iterations to declare convergence.
        """
        if k < 1:
            raise ValueError('K must be a positive number')
        
        self.k = k
        self.max_it = max_it
        self.seed = seed
        self.tol = tol
        self.n_it = 0
        self.re_dis = 1
        self.metric = 'euclidean'
        self.cost_function_type = 'mean'
        self.previous_centroids = None
        self.X = None
        self.centroids = None
        self.nearest = None
        self.inertia = None
        
    def fit(self, X: np.ndarray):
        """
        Compute cluster centers and predict cluster index for each sample.
        :param X: 2D data array of size (rows, features).
        """
        np.random.seed(self.seed)
        self.X = X
        sse = []
        # Initialize centroids
        init_centroids = np.random.choice(range(self.X.shape[0]), size=self.k, replace=False)
        self.centroids = self.X[init_centroids, :]
        
        
        while self.re_dis > self.tol and self.n_it < self.max_it:
            self.re_dis = 0
            self.n_it += 1
            
            distances = self._distances(X)
            labels, self.nearest, nearest_ids = self._get_nearest(X, distances)
            self.previous_centroids = self.centroids.copy()
            self._shift_centroids()
            self.re_dis = self._calculate_cost()

        # inertia calculus
        np_labels = np.array(labels)
        for act in range(0, self.k):
            index = np.where(np_labels == act)[0]
            cluster_x = self.X[index, :]
            se = np.sum(np.linalg.norm(cluster_x - self.centroids[act]))
            sse.append(se)
        self.inertia = np.min(sse)



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

    def _calculate_cost(self):
        """
        Calculate the distance between old centroids and new ones (SD).
        :return: SD.
        """
        cost = 0
        for k in range(self.k):
            cost+= distance.cdist(np.array([self.centroids[k]]), np.array([self.previous_centroids[k]]), metric=self.metric)[0][0]
        return cost


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

    def _shift_centroids(self):
        """Compute the new centroid for each cluster using method depending on algorithm."""
        for k in range(self.k):
            if len(self.nearest[k]) > 0:
                if self.cost_function_type =="median":
                    self.centroids[k, :] = np.median(np.array(self.nearest[k]), axis=0)
                elif self.cost_function_type == "mean":
                    self.centroids[k, :] = np.mean(np.array(self.nearest[k]), axis=0)
        
    