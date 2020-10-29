from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


def preprocess():
    # load dataset
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "datasets", "breast-w.arff")
    data, meta = arff.loadarff(f)
    dataset = np.array(data.tolist(), dtype=object)
    meta = meta.names()  # list containing column names

    # create a initial pandas dataframe
    df = pd.DataFrame(data=dataset, columns=list(meta))

    # detect missing values and replacing using median
    if df.isnull().any().sum() > 0:
        for x in meta[:-1]:
            median = df[x].median()
            df[x].fillna(median, inplace=True)

    # split-out dataset
    x = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    def label_encoding(value, classes: bytearray):
        """Function written based on label encoding rule
            Parameters
            ----------
            value : value to interpret
            classes: array of classes
            Returns
            -------
            self : returns index of x value in classes array.
            """
        return classes.index(value)

    label_true = list(y.iloc[:].apply(label_encoding, classes=[b'malignant', b'benign']))

    # fit MinMax scaler on data
    norm = MinMaxScaler().fit(x)
    # transform data
    x = norm.transform(x)

    return x, label_true, df
