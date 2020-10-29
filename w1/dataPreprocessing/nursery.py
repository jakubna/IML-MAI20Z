from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


def preprocess():
    # load dataset
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "datasets", "nursery.arff")
    data, meta = arff.loadarff(f)
    dataset = np.array(data.tolist(), dtype=object)

    column_names = meta.names()  # list containing column names
    # create a initial pandas dataframe
    df = pd.DataFrame(data=dataset, columns=column_names)

    # detect and replace missing values
    if df.isnull().any().sum() > 0:
        for x in meta[:-1]:
            top_frequent_value = df[x].describe()['top']
            df[x].fillna(top_frequent_value, inplace=True)

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

    label_true = list(y.iloc[:].apply(label_encoding, classes=[b'not_recom', b'recommend', b'very_recom',
                                                               b'priority', b'spec_prior']))

    # apply label_encoding function on selected columns
    x.a2 = x.a2.apply(label_encoding, classes=[b'proper', b'less_proper', b'improper', b'critical',
                                               b'very_crit'])
    x.a4 = x.a4.apply(label_encoding, classes=[b'1', b'2', b'3', b'more'])
    x.a5 = x.a5.apply(label_encoding, classes=[b'convenient', b'less_conv', b'critical'])
    x.a6 = x.a6.apply(label_encoding, classes=[b'convenient', b'inconv'])
    x.a7 = x.a7.apply(label_encoding, classes=[b'nonprob', b'slightly_prob', b'problematic'])
    x.a8 = x.a8.apply(label_encoding, classes=[b'not_recom', b'recommended', b'priority'])

    # apply one-hot encoding using get_dummies function from pandas
    x = pd.get_dummies(x, columns=['a1', 'a3'])

    # fit MinMax scaler on data
    norm = MinMaxScaler().fit(x)
    # transform data
    x = norm.transform(x)

    return x, label_true, df
