from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from dataPreprocessing.utils import label_encoding
import os

def preprocess():
    """
    Apply the personalized operations to preprocess the database.
    :return: dict:
            db: 2D data array of size (rows, features),
            label_true: array of true label values,
            data_frame: raw data set with filled missing values in.
    """
    # load dataset
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "datasets", "adult.arff")
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

    # split-out dataset
    X = df.iloc[:, :-1].copy()
    Y = df.iloc[:, -1].copy()
    # name of columns with numerical features
    numerical_features = ['age',
                          'fnlwgt',
                          'education-num',
                          'capital-gain',
                          'capital-loss',
                          'hours-per-week']
    #name of columns with categorical features
    categorical_features = ['workclass',
                            'education',
                            'marital-status',
                            'occupation',
                            'relationship',
                            'race',
                            'sex',
                            'native-country']
    # create instance of one hot encoder
    onehotencoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # transform categorical data
    X_categorical = onehotencoder.fit_transform(X[categorical_features])
    columns = onehotencoder.get_feature_names(input_features=categorical_features)
    X_categorical = pd.DataFrame(data=X_categorical, columns=columns)
    # drop redundant column
    X_categorical = X_categorical.drop(columns=["sex_b'Male'"])

    # fit MinMax scaler on data
    norm = MinMaxScaler()
    # transform numerical data
    X_numerical = norm.fit_transform(X[numerical_features])
    X_numerical = pd.DataFrame(data=X_numerical, columns=numerical_features)
    # concatenate final preprocessed data set
    X_preprocessed = pd.concat((X_categorical, X_numerical), axis=1)

    # get the true labels values of the dataset
    label_true = list(Y.iloc[:].apply(label_encoding, classes=[b'>50K', b'<=50K']))

    return (dict(db=X_preprocessed.to_numpy(), label_true=label_true, data_frame=df))



