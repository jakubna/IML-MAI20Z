import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from dataPreprocessing.utils import label_encoding as label_encoding
from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF


def preprocess(dataset_train, dataset_val, meta, weights, n_neigh):
    """
    Apply the personalized operations to preprocess the database.
    :return: dict:
            db: 2D data array of size (rows, features),
            label_true: array of true label values,
            data_frame: raw data set with filled missing values in.
    """
    if weights not in ['mutual_info', 'relief', 'equal']:
        raise ValueError('Param weights can be: equal, relief, mutual_info or correlation')

    column_names = meta.names()  # list containing column names

    # create a initial pandas dataframe
    df_train = pd.DataFrame(data=dataset_train, columns=column_names)
    df_val = pd.DataFrame(data=dataset_val, columns=column_names)

    # split-out dataset
    X_train = df_train.iloc[:, :-1].copy()
    Y_train = df_train.iloc[:, -1].copy()

    X_val = df_val.iloc[:, :-1].copy()
    Y_val = df_val.iloc[:, -1].copy()

    # get the true labels values of the dataset
    y_train = list(Y_train.iloc[:].apply(label_encoding, classes=[b'black', b'white']))
    y_val = list(Y_val.iloc[:].apply(label_encoding, classes=[b'black', b'white']))

    # fit MinMax scaler on data
    norm = MinMaxScaler().fit(X_train)
    # transform data
    X_norm_train = norm.transform(X_train)
    X_norm_val = norm.transform(X_val)

    wei = weights_values(X_norm_train, y_train, n_neigh, weights)

    X_preprocessed_train = X_norm_train * wei
    X_preprocessed_val = X_norm_val * wei

    return (X_preprocessed_train, y_train), (X_preprocessed_val, y_val)


def weights_values(x: np.ndarray, y: np.array, n_neigh, weights):
    """
    Create the weights vector for the problem
    :param x: 2D data array of size (rows, features).
    :param y: data array of size (rows).
    :param n_neigh: number of neighbors for knn
    :param weights: policy of weights that we want to apply
    Returns: weights vector in numpy format
    """
    if weights == 'equal':
        return np.ones(x.shape[1])
    if weights == 'relief':
        fs = ReliefF(n_neighbors=n_neigh, n_features_to_keep=x.shape[1])
        fs.fit(x, np.array(y))
        rel = fs.feature_scores
        return rel / np.linalg.norm(rel)
    if weights == 'mutual_info':
        w = np.array(mutual_info_classif(x, y, n_neighbors=n_neigh))
        return w / np.linalg.norm(w)
    raise ValueError('Param weights can be: equal, relief, mutual_info')
