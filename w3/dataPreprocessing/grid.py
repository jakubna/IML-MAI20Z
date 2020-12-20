import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from dataPreprocessing.utils import label_encoding as label_encoding


def preprocess(dataset_train, dataset_val, meta):
    """
    Apply the personalized operations to preprocess the database.
    :return: dict:
            db: 2D data array of size (rows, features),
            label_true: array of true label values,
            data_frame: raw data set with filled missing values in.
    """

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

    X_preprocessed_train = X_norm_train
    X_preprocessed_val = X_norm_val

    return (X_preprocessed_train, y_train), (X_preprocessed_val, y_val)
