from numpy import linalg as LA
import numpy as np
from scipy.spatial import distance
from collections import Counter


class kNNAlgorithm:

    def __init__(self, n_neighbors: int = 5, weights='majority_class', metric='minkowski'):
        """
        :param n_neighbors: Number of neighbors to use by default for kneighbors queries.
        :param weights: Weight function used in prediction. Possible values: 'majority_class', 'inverse_distance','sheppard_work'.
        :param metric: The distance metric to use for the tree
        """
        if n_neighbors < 1:
            raise ValueError('n_neighbors must be a positive number')
        if weights not in ['majority_class', 'inverse_distance,'sheppard_work']:
            raise ValueError('Param weights can be: uniform or distance')
        if metric not in ['minkowski', 'euclidean']:
            raise ValueError('Param metric can be: minkowski or euclidean')

        self.n_neighbors = n_neighbors
        self.weights = weights
        if metric == "minkowski":
            self.metric = 'cityblock'
        else:
            self.metric = metric

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def fit(self, X: np.ndarray, y):
        """
        Fit the model using X as training data and y as target values
        :param X: 2D data array of size (rows, features).
        """
        self.X_train = X
        self.y_train = y

    def kneighbors(self, X: np.ndarray, return_distance=False):
        """
        Finds the K-neighbors of a point.
        :param X: 2D data array of size (rows, features).
        :param return_distance: If False, distances will not be returned.
        """
        self.X_test = X
        neigh_dist = []
        neigh_ind = []

        point_dist = self._calculate_distance(self.X_test, self.X_train)

        for point in list(point_dist):
            enum_neigh = enumerate(point)
            sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:self.n_neighbors]

            ind_list = [tup[0] for tup in sorted_neigh]
            dist_list = [tup[1] for tup in sorted_neigh]

            neigh_dist.append(dist_list)
            neigh_ind.append(ind_list)

        if return_distance:
            return np.array(neigh_dist), np.array(neigh_ind)

        return np.array(neigh_ind)

    def predict(self, X: np.ndarray):
        """
        Predict the class labels for the provided data.
        :param X: 2D data array of size (rows, features).
        Three policies can be used: majority_class, inverse_distance, sheppard_work
        'majority_class' : uniform weights. All points in each neighborhood are weighted equally.
        'inverse_distance' : weight points by the inverse of their distance. in this case,
            closer neighbors of a query point will have a greater influence than neighbors which are further away.
        'sheppard_work' : uses an exponential function rather than the inverse distance.
        """
        self.X_test = X
        y_train = np.array(self.y_train)
        if self.weights == "majority_class":
            neigh_ind = self.kneighbors(self.X_test)
            y_pred = np.array([np.argmax(np.bincount(y_train[neigh])) for neigh in neigh_ind])
            return y_pred

        elif self.weights == "inverse_distance":
            class_counter = Counter()
            y_pred=[]
            neigh_dist, neigh_ind = self.kneighbors(self.X_test, return_distance=True)
            for n in range(len(neigh_ind)):
                for index in range(len(n)):
                    dist=neigh_dist[n][index]
                    label=neigh_ind[n][index]
                    class_counter[label]=+ 1/dist
                y_pred.append(class_counter.most_common(1)[0][0])
             return y_pred
         elif self.weights == "sheppard_work":
            class_counter = Counter()
            y_pred=[]
            neigh_dist, neigh_ind = self.kneighbors(self.X_test, return_distance=True)
            for n in range(len(neigh_ind)):
                for index in range(len(n)):
                    dist=neigh_dist[n][index]
                    label=neigh_ind[n][index]
                    class_counter[label]=+ exp(-dist)
                y_pred.append(class_counter.most_common(1)[0][0])
             return y_pred                   

    def _calculate_distance(self, x: np.ndarray, y: np.ndarray):
        """
        Calculate distance between 2 elements using the metric depending on the algorithm ('euclidean' or 'cityblock').
        :param x: 1D vector with all x attributes.
        :param y: 1D vector with all y attributes.
        :return: Distance between both vectors using the specified metric.
        """
        return distance.cdist(np.array(x), np.array(y), metric=self.metric)
