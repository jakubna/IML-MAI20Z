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

        distances = KMeans._distances(X)
        labels, self.nearest, nearest_ids = KMeans._get_nearest(X, distances)

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

            

