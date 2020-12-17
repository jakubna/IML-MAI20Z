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
    # load dataset
    column_names = meta.names()  # list containing column names
    column_types = meta.types()  # list containing column types
    column_types_dict = dict(zip(column_names[:-1], column_types[:-1]))  # dictionary containing column names and types

    # create a initial pandas dataframe
    df_train = pd.DataFrame(data=dataset_train, columns=column_names)
    df_val = pd.DataFrame(data=dataset_val, columns=column_names)

    # detect and replace missing values
    if df_train.isnull().any().sum() > 0 or df_val.isnull().any().sum() > 0:
        for x in meta[:-1]:
            if column_types_dict[x] == 'nominal':
                top_frequent_value = df_train[x].describe()['top']
                df_train[x].fillna(top_frequent_value, inplace=True)
                df_val[x].fillna(top_frequent_value, inplace=True)
            else:
                median = df_train[x].median()
                df_train[x].fillna(median, inplace=True)
                df_val[x].fillna(median, inplace=True)

    # split-out dataset
    X_train = df_train.iloc[:, :-1].copy()
    Y_train = df_train.iloc[:, -1].copy()

    X_val = df_val.iloc[:, :-1].copy()
    Y_val = df_val.iloc[:, -1].copy()

    # name of columns with numerical features
    numerical_features = ['age',
                          'fnlwgt',
                          'education-num',
                          'capital-gain',
                          'capital-loss',
                          'hours-per-week']
    # name of columns with categorical features
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
    X_categorical_train = onehotencoder.fit_transform(X_train[categorical_features])
    X_categorical_val = onehotencoder.transform(X_val[categorical_features])

    columns = onehotencoder.get_feature_names(input_features=categorical_features)

    X_categorical_train = pd.DataFrame(data=X_categorical_train, columns=columns)
    X_categorical_val = pd.DataFrame(data=X_categorical_val, columns=columns)

    # drop redundant column
    X_categorical_train = X_categorical_train.drop(columns=["sex_b'Male'"])
    X_categorical_val = X_categorical_val.drop(columns=["sex_b'Male'"])

    # fit MinMax scaler on data
    norm = MinMaxScaler()

    # transform numerical data
    X_numerical_train = norm.fit_transform(X_train[numerical_features])
    X_numerical_val = norm.transform(X_val[numerical_features])

    X_numerical_train = pd.DataFrame(data=X_numerical_train, columns=numerical_features)
    X_numerical_val = pd.DataFrame(data=X_numerical_val, columns=numerical_features)

    # concatenate final preprocessed data set
    X_preprocessed_train = pd.concat((X_categorical_train, X_numerical_train), axis=1)
    X_preprocessed_val = pd.concat((X_categorical_val, X_numerical_val), axis=1)

    # get the true labels values of the dataset
    y_train = list(Y_train.iloc[:].apply(label_encoding, classes=[b'>50K', b'<=50K']))
    y_val = list(Y_val.iloc[:].apply(label_encoding, classes=[b'>50K', b'<=50K']))

    return (X_preprocessed_train.to_numpy(), y_train), (X_preprocessed_val.to_numpy(), y_val)




