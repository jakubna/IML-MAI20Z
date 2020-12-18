import numpy as np
from scipy.spatial import distance
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF


class kNNAlgorithm:

    def __init__(self, n_neighbors: int = 5, weights='None', policy='majority_class', metric='minkowski'):
        """
        :param n_neighbors: Number of neighbors to use by default for kneighbors queries.
        :param policy: policy dor the algorithm used in prediction. Possible values: 'majority_class',
                        'inverse_distance','sheppard_work'.
        :param weights: weights policy of the function. Possible values: 'equal', 'mutual_info', 'relief'.
        :param metric: The distance metric to use for the tree
        """
        if n_neighbors < 1:
            raise ValueError('n_neighbors must be a positive number')
        if policy not in ['majority_class', 'inverse_distance', 'sheppard_work']:
            raise ValueError('Param policy can be: uniform or distance')
        if metric not in ['minkowski', 'euclidean', 'chebyshev', 'canberra']:
            raise ValueError('Param metric can be: minkowski or euclidean or chebyshev or canberra')
        if weights not in ['mutual_info', 'relief', 'equal']:
            raise ValueError('Param weights can be: equal, relief, mutual_info or correlation')

        self.n_neighbors = n_neighbors
        self.policy = policy
        self.weights = weights
        if metric == "minkowski":
            self.metric = 'cityblock'
        else:
            self.metric = metric

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def fit(self, x: np.ndarray, y):
        """
        Fit the model using X as training data and y as target values
        :param x: 2D data array of size (rows, features).
        :param y: true labels
        """
        self.X_train = x
        self.y_train = y

    def kneighbors(self, x: np.ndarray, return_distance=False):
        """
        Finds the K-neighbors of a point.
        :param x: 2D data array of size (rows, features).
        :param return_distance: If False, distances will not be returned.
        """
        self.X_test = x
        neigh_dist = []
        neigh_ind = []

        point_dist = self._calculate_distance(self.X_test, self.X_train)

        for point in list(point_dist):
            enum_neigh = enumerate(point)
            sorted_neigh = sorted(enum_neigh, key=lambda z: z[1])[:self.n_neighbors]

            ind_list = [tup[0] for tup in sorted_neigh]
            dist_list = [tup[1] for tup in sorted_neigh]

            neigh_dist.append(dist_list)
            neigh_ind.append(ind_list)

        if return_distance:
            return np.array(neigh_dist), np.array(neigh_ind)

        return np.array(neigh_ind)

    def predict(self, x: np.ndarray):
        """
        Predict the class labels for the provided data.
        :param x: 2D data array of size (rows, features).
        Three policies can be used: majority_class, inverse_distance, sheppard_work
        'majority_class' : uniform weights. All points in each neighborhood are weighted equally.
        'inverse_distance' : weight points by the inverse of their distance. in this case,
            closer neighbors of a query point will have a greater influence than neighbors which are further away.
        'sheppard_work' : uses an exponential function rather than the inverse distance.
        """
        self.X_test = self.weights_values(x)
        y_train = np.array(self.y_train)
        if self.policy == "majority_class":
            neigh_ind = self.kneighbors(self.X_test)
            y_pred = np.array([np.argmax(np.bincount(y_train[neigh])) for neigh in neigh_ind])
            return y_pred

        elif self.policy == "inverse_distance":
            class_counter = Counter()
            y_pred = []
            neigh_dist, neigh_ind = self.kneighbors(self.X_test, return_distance=True)
            for n, neigh in enumerate(neigh_ind):
                for index,nneigh in enumerate(neigh):
                    dist = neigh_dist[n][index]
                    label = y_train[neigh_ind[n][index]]
                    if dist ==0:
                        dist=0.000001
                    class_counter[label] = + 1 / dist
                y_pred.append(class_counter.most_common(1)[0][0])
            return y_pred
        elif self.policy == "sheppard_work":
            class_counter = Counter()
            y_pred = []
            neigh_dist, neigh_ind = self.kneighbors(self.X_test, return_distance=True)
            for n, neigh in enumerate(neigh_ind):
                for index,nneigh in enumerate(neigh):
                    dist = neigh_dist[n][index]
                    label = y_train[neigh_ind[n][index]]
                    class_counter[label] = + np.exp(-dist)
                y_pred.append(class_counter.most_common(1)[0][0])
            return y_pred

    def _calculate_distance(self, x: np.ndarray, y: np.ndarray):
        """
        Calculate distance between 2 elements using the metric depending on the algorithm 
        ('euclidean', 'cityblock', 'canberra', 'chebyshev').
        :param x: 1D vector with all x attributes.
        :param y: 1D vector with all y attributes.
        :return: Distance between both vectors using the specified metric.
        """
        if self.metric == 'euclidean' or self.metric == 'cityblock':
            return distance.cdist(np.array(x), np.array(y), metric=self.metric)
        elif self.metric == 'canberra':
            distances = np.zeros((x.shape[0], y.shape[0]))
            for i in range(x.shape[0]):
                for j in range(y.shape[0]):
                    distances[i, j] = self._canberra_distance(x[i], y[j])
            return distances
        elif self.metric == 'chebyshev':
            distances = np.zeros((x.shape[0], y.shape[0]))
            for i in range(x.shape[0]):
                for j in range(y.shape[0]):
                    distances[i, j] = self._chebyshev_distance(x[i], y[j])
            return distances

    def _chebyshev_distance(self, p_vec, q_vec):
        """
        This method implements the Chebyshev distance metric
        :param p_vec: vector one
        :param q_vec: vector two
        :return: the Chebyshev distance between vector one and two
        """
        return max(np.fabs(p_vec - q_vec))

    def _canberra_distance(self, p_vec, q_vec):
        """
        This method implements the canberra distance metric
        :param p_vec: vector one
        :param q_vec: vector two
        :return: the canberra distance between vector one and two
        """
        return np.sum(np.nan_to_num(np.fabs(p_vec - q_vec) / (np.fabs(p_vec) + np.fabs(q_vec))))

    def weights_values(self, x: np.ndarray):
        """
        Create the weights vector for the problem
        :param x: 2D data array of size (rows, features).
        Returns: weights vector in numpy format
        """
        if self.weights == 'equal':
            return x
        if self.weights == 'relief':
            fs = ReliefF(n_neighbors=self.n_neighbors, n_features_to_keep=x.shape[1])
            return np.array(fs.fit_transform(x, np.array(self.y_train)))
        if self.weights == 'mutual_info':
            return x * np.array(mutual_info_classif(x, self.y_train, n_neighbors=self.n_neighbors))
        raise ValueError('Param weights can be: equal, relief, mutual_info')
