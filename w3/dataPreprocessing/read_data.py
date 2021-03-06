import os
from scipy.io import arff
import numpy as np
from dataPreprocessing.hypothyroid import preprocess as preprocess_hypothyroid
from dataPreprocessing.grid import preprocess as preprocess_grid


def read_database(database: str):
    """
    With the input string choose the dataset that we want to execute and call preprocess function.
    :param database: string with the name of the dataset that we want to execute.
    :return: features of the database(processed database, true classification results, complete dataframe).
    """
    name = database
    folds = []

    if name not in ['hypothyroid', 'grid']:
        raise ValueError('Database not found')
    if name == "hypothyroid":
        preprocess = preprocess_hypothyroid
    elif name == "grid":
        preprocess = preprocess_grid

    for i in range(10):
        fold_path = name + ".fold.00000" + str(i)
        f_train = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", name, fold_path + ".train.arff")
        f_val = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", name, fold_path + ".test.arff")
        data_train, meta_train = arff.loadarff(f_train)
        data_val, meta_val = arff.loadarff(f_val)
        dataset_train = np.array(data_train.tolist(), dtype=object)
        dataset_val = np.array(data_val.tolist(), dtype=object)
        (X_train, y_train), (X_val, y_val) = preprocess(dataset_train, dataset_val, meta_train)
        folds.append({
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        })
    return folds
