from dataPreprocessing.adult import preprocess as preprocess_adult
from evaluation.plot import *
from evaluation.evaluate import *
import pandas as pd
from algorithms.kNNAlgorithm import kNNAlgorithm
import itertools
import time


def best_knn(x: np.ndarray, label_true):
    """
    Apply the implemented algorithms, dbscan and evaluate the obtained results.
    :param x: data array of size (n_folds), each element is a matrix that represent the elements in one fold.
    :param label_true: data array of size (n_folds), each element is one fold results solution.
    """

    # params = dict(n_neighbors=[1, 3, 5, 7], weights=['equal', 'mutual_info', 'relief'],
    #              policy=['majority_class', 'inverse_distance', 'sheppard_work'], metric=['minkowski', 'euclidean'])
    params = [[1, 3, 5, 7], ['equal', 'mutual_info', 'relief'], ['majority_class', 'inverse_distance', 'sheppard_work'],
              ['minkowski', 'euclidean']]

    # get all combinations that must be evaluated
    pos_sol = np.array(list(itertools.product(*params)))
    full_results = []

    for act in pos_sol:
        # get our KNN
        av_results = dict(metrics=act, accuracy=0, precision=0, recall=0, f1score=0, time=0)
        knn = kNNAlgorithm(n_neighbors=act[0], weights=act[1], policy=act[2], metric=act[3])
        for i in range(x.shape[0]):
            x_train, y_train, x_validate, y_validate = get_current_xy_matrix(x, label_true, i)
            t0 = time.time()
            knn.fit(x_train, y_train)
            predict = knn.predict(x_validate)
            t1 = time.time() - t0

            # evaluate the knn
            supervised = evaluate_supervised_external(y_validate, predict)

            # sum the new metrics obtained to make average
            metrics = ['accuracy', 'precision', 'recall', 'f1score']
            for m in metrics:
                av_results[m] += supervised[m]
            av_results['time'] = t1

        # make the average
        metrics = ['accuracy', 'precision', 'recall', 'f1score', 'time']
        for m in metrics:
            av_results[m] = av_results[m] / 10

        full_results.append(av_results)
    set_output(full_results, "all_knn_metrics_cases")
    # find the best one


def get_current_xy_matrix(x: np.ndarray, y: np.ndarray, i):
    """
    Get the current x_train, y_train, x_validate, y_validate for this cross validation loop
    :param x: data array of size (n_folds), each element is a matrix that represent the elements in one fold.
    :param y: data array of size (n_folds), each element is one fold results solution.
    :param i: current validation fold index in x array.
    :return: x_train, y_train, x_validate, y_validate.
    """
    neo_x_val = np.ndarray(x[i])
    neo_y_val = np.ndarray(y[i])
    neo_x = None
    neo_y = None
    for a in range(x.shape[0]):
        if a != i and neo_x is None:
            neo_x = x[a]
            neo_y = y[a]
        if a != i and neo_x is not None:
            neo_x = np.concatenate((neo_x, x[a]))
            neo_y = np.concatenate(neo_y, y[a])
    return neo_x, neo_y, neo_x_val, neo_y_val


def apply_evaluation(x, label_true, labels, names, database_name):
    """
    Apply all the evaluations to the implemented algorithms and classify in a dataframe.
    :param x: 2D data array of size (rows, features).
    :param label_true: labels of the real classification extracted from the database.
    :param labels: predicted labels.
    :param names: list of all the evaluated algorithms.
    :param database_name: name of the database that is being tested
    :return: a dataframe with the evaluation results for algorithms implemented in this practise.
    """
    rows = []

    for i in range(0, len(names)):
        act_name = names[i]
        act_data = x[i]

        # unsupervised = evaluate_unsupervised_internal(act_data, labels)
        supervised = evaluate_supervised_external(label_true, labels)

        row = {**dict(Names=act_name), **supervised}  # , **unsupervised
        rows.append(row)
    df_results = pd.DataFrame(rows)
    set_output(df_results, 'knn_analysis_'+database_name)


def preprocess_database(database: str):
    """
    With the input string choose the dataset that we want to execute and call preprocess function.
    :param database: string with the name of the dataset that we want to execute.
    :return: features of the preprocessed database(processed database, true classification results, complete dataframe).
    """
    # processed -> db, label_true, data_frame
    if database == "adult":
        processed = preprocess_adult()
    else:
        raise ValueError('database not found')

    return processed


def split_db_original(x,  components):
    """
    Method that process input database with the row chosen by user.
    :param x: processed dataset 2D numpy array.
    :param components: names and index of the features chosen by user to plot for the original data set plot.
    """
    cm = components[1]
    ap = []
    for itera in cm:
        ap.append(x[:, itera].tolist())
    ap_np = np.transpose(np.array(ap))

    return ap_np


def get_features(data_frame, n_components):
    """
    Function that ask to the user which features want to see in the plot of the original data set.
    :param data_frame: original dataframe.
    :param n_components: number of components specified by user
    :return: the names of the features that user choose and its index in the matrix.
    """
    n_features = 3
    if n_components == 2:
        n_features = 2
    col = data_frame.columns.tolist()[:-1]
    com = 1
    components = []
    index = []
    for n_iter in range(n_features):
        print("Choose the {}-feature that you want to plot: ".format(n_iter + 1))
        for i in range(len(col)):
            if col[i] == -1:
                print('\033[91m'" {}-> SELECTED \033[0m".format(i + 1))
            else:
                print("{} -> {}".format(i + 1, col[i]))

        try:
            com = int(input("write the left index of the feature: "))
        except:
            print('Invalid age, please enter a number')
        components.append(col[com - 1])
        index.append(com - 1)
        col[com - 1] = -1

    return components, index


def set_output(results, database_name):
    # print result at the terminal
    print(results)

    # load results to a csv file
    results.to_csv("./results/" + database_name + ".csv", sep='\t', encoding='utf-8', index=False)

    print("\nThe CSV output files are created in results folder of this project\n")

