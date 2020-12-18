import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from dataPreprocessing.utils import label_encoding as label_encoding
from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF


def preprocess(dataset_train, dataset_val, meta, weights, n_neigh):
    """
    Apply the personalized operations to preprocess the database.
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
        for x in column_names[:-1]:
            if column_types_dict[x] == 'nominal':
                top_frequent_value = df_train[x].describe()['top']
                df_train[x].fillna(top_frequent_value, inplace=True)
                df_val[x].fillna(top_frequent_value, inplace=True)
            else:
                if x != "TBG":
                    median = df_train[x].median()
                    df_train[x].fillna(median, inplace=True)
                    df_val[x].fillna(median, inplace=True)

    # split-out dataset
    X_train = df_train.iloc[:, :-1].copy()
    Y_train = df_train.iloc[:, -1].copy()

    X_val = df_val.iloc[:, :-1].copy()
    Y_val = df_val.iloc[:, -1].copy()

    # name of columns with numerical features (we omit TBG feature, cause all data = Nan)
    numerical_features = ['age',
                          'TSH',
                          'T3',
                          'TT4',
                          'T4U',
                          'FTI']
    # name of columns with categorical features
    categorical_features = ['sex',
                            'on_thyroxine',
                            'query_on_thyroxine',
                            'on_antithyroid_medication',
                            'sick',
                            'pregnant',
                            'thyroid_surgery',
                            'I131_treatment',
                            'query_hypothyroid',
                            'query_hyperthyroid',
                            'lithium',
                            'goitre',
                            'hypopituitary',
                            'psych',
                            'TSH_measured',
                            'T3_measured',
                            'TT4_measured',
                            'T4U_measured',
                            'FTI_measured',
                            'TBG_measured',
                            'referral_source']
    # create instance of one hot encoder
    onehotencoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # transform categorical data
    X_categorical_train = onehotencoder.fit_transform(X_train[categorical_features])
    X_categorical_val = onehotencoder.transform(X_val[categorical_features])

    columns = onehotencoder.get_feature_names(input_features=categorical_features)

    X_categorical_train = pd.DataFrame(data=X_categorical_train, columns=columns)
    X_categorical_val = pd.DataFrame(data=X_categorical_val, columns=columns)

    # drop redundant column
    columns_to_drop = ["on_thyroxine_b'f'",
                       "query_on_thyroxine_b'f'",
                       "on_antithyroid_medication_b'f'",
                       "sick_b'f'",
                       "pregnant_b'f'",
                       "thyroid_surgery_b'f'",
                       "I131_treatment_b'f'",
                       "query_hypothyroid_b'f'",
                       "query_hyperthyroid_b'f'",
                       "lithium_b'f'",
                       "goitre_b'f'",
                       "hypopituitary_b'f'",
                       "psych_b'f'",
                       "TSH_measured_b'f'",
                       "T3_measured_b'f'",
                       "TT4_measured_b'f'",
                       "T4U_measured_b'f'",
                       "FTI_measured_b'f'"]
    X_categorical_train = X_categorical_train.drop(columns=columns_to_drop)
    X_categorical_val = X_categorical_val.drop(columns=columns_to_drop)

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

    X_norm_val = X_preprocessed_val.to_numpy()
    X_norm_train = X_preprocessed_train.to_numpy()

    # get the true labels values of the dataset
    y_train = list(Y_train.iloc[:].apply(label_encoding,
                                         classes=[b'negative', b'compensated_hypothyroid', b'primary_hypothyroid',
                                                  b'secondary_hypothyroid']))
    y_val = list(Y_val.iloc[:].apply(label_encoding,
                                     classes=[b'negative', b'compensated_hypothyroid', b'primary_hypothyroid',
                                              b'secondary_hypothyroid']))

    # apply weights policies
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
        rel = np.array(fs.fit(x, y).feature_scores)
        return rel / np.linalg.norm(rel)
    if weights == 'mutual_info':
        w = np.array(mutual_info_classif(x, y, n_neighbors=n_neigh))
        return w / np.linalg.norm(w)
    raise ValueError('Param weights can be: equal, relief, mutual_info')
