from dataPreprocessing.adult import preprocess as preprocess_adult
from evaluation.plot import *
from evaluation.evaluate import *
import pandas as pd
from algorithms.kNNAlgorithm import kNNAlgorithm
import itertools
import time


def best_knn_metrics(k_x: np.ndarray, name_file):
    """
    Apply the implemented knn for each possible parameter combination, extract the metrics and store to a csv file.
    :param k_x: data array of size (n_folds), each element is a dictionary with X_train, y_train, X_val, y_val.
    :param name_file: the name of the file on where you want to write
    """

    params = [[1, 3, 5, 7], ['equal', 'mutual_info', 'relief'], ['majority_class', 'inverse_distance', 'sheppard_work'],
              ['minkowski', 'euclidean', 'chebyshev']]  # , 'chebyshev' 'canberra'

    # get all combinations that must be evaluated
    pos_sol = np.array(list(itertools.product(*params)))
    print("Number of possible combinations:", pos_sol.shape[0])
    pos_sol = [pos_sol[3]]
    full_results = []

    for act in pos_sol:
        # get our KNN
        print(act)
        av_results = dict(metrics=act, av_accuracy=0, av_time=0, accuracy=[], time=[])
        knn = kNNAlgorithm(n_neighbors=int(act[0]), weights=act[1], policy=act[2], metric=act[3])
        for act_fold in k_x:
            x_train = act_fold['X_train']
            y_train = act_fold['y_train']
            x_validate = act_fold['X_val']
            y_validate = act_fold['y_val']
            t0 = time.time()
            knn.fit(x_train, y_train)
            predict = knn.predict(x_validate)
            t1 = time.time() - t0

            # evaluate the knn
            supervised = evaluate_accuracy(y_validate, predict)

            # sum the new metrics obtained to make average
            av_results['accuracy'].append(supervised['accuracy'])
            av_results['time'].append(t1)

        # make the average results
        av_results['accuracy'] = np.array(av_results['accuracy'])
        av_results['time'] = np.array(av_results['time'])
        av_results['av_accuracy'] = np.average(av_results['accuracy'])
        av_results['av_time'] = np.average(av_results['time'])

        full_results.append(av_results)
    df_results = pd.DataFrame(full_results)
    set_output(df_results, name_file)
    # find the best one


def best_knn_get_best_comb(name_file):
    """
    Apply the evaluation of the metrics extracted for each combination of parameters.
    :param name_file: the name of the file on where you want to read the metrics.
    """
    metrics = read_csv(name_file)
    print(type(metrics))
    print(metrics)


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


def read_csv(name_file):
    """
    Read the csv file and extract the metrics for each combination.
    :param name_file: the name of the file on where you want to read the metrics.
    """
    results = pd.read_csv("./results/" + name_file + ".csv", sep='\t', encoding='utf-8')
    return np.array(results.to_dict(orient='records'))

