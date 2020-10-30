from scipy.io import arff
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


def preprocess():
    """
    Apply the personalized operations to preprocess the database.
    :return: 2D data array of size (rows, features).
    """
    # load dataset
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "datasets", "cmc.arff")
    data, meta = arff.loadarff(f)
    dataset = np.array(data.tolist(), dtype=object)

    column_names = meta.names()  # list containing column names
    column_types = meta.types()  # list containing column types
    column_types_dict = dict(zip(column_names[:-1], column_types[:-1]))  # dictionary containing column names and types

    # create a initial pandas dataframe
    df = pd.DataFrame(data=dataset, columns=column_names)

    # detect and replace missing values
    if df.isnull().any().sum() > 0:
        for x in meta[:-1]:
            if column_types_dict[x] == 'nominal':
                top_frequent_value = df[x].describe()['top']
                df[x].fillna(top_frequent_value, inplace=True)
            else:
                median = df[x].median()
                df[x].fillna(median, inplace=True)

    # create instance of label_encoder
    label_encoder = LabelEncoder()
    # transform data
    for k, v in column_types_dict.items():
        if v == 'nominal':
            label_encoder.fit(sorted(df[k].unique()))
            df[k] = label_encoder.transform(df[k])

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

    # get the true labels values of the dataset
    label_true = list(y.iloc[:].apply(label_encoding, classes=[b'1', b'2', b'3']))

    # fit MinMax scaler on data
    norm = MinMaxScaler().fit(x)
    # transform data
    x = norm.transform(x)

    return dict(db=x, label_true=label_true, data_frame=df)

