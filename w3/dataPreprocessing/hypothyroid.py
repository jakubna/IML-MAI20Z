from dataPreprocessing.hypothyroid import preprocess as preprocess_hypothyroid
from dataPreprocessing.grid import preprocess as preprocess_grid
from evaluation.plot import *
from evaluation.evaluate import *
import pandas as pd
from algorithms.kNNAlgorithm import kNNAlgorithm
import itertools
import time


def best_knn_metrics(k_x: np.ndarray, name_file, name_db):
    """
    Apply the implemented knn for each possible parameter combination, extract the metrics and store to a csv file.
    :param k_x: data array of size (n_folds), each element is a dictionary with validation_data, train_data, meta_data.
    :param name_file: the name of the file on where you want to write.
    :param name_db: string with the name of our dataset.
    """
    params = [[1, 3, 5, 7], ['equal', 'mutual_info', 'relief'], ['majority_class', 'inverse_distance', 'sheppard_work'],
              ['minkowski', 'euclidean', 'chebyshev']]  # , 'chebyshev' 'canberra'

    # get all combinations that must be evaluated
    pos_sol = np.array(list(itertools.product(*params)))
    print("Number of possible combinations:", pos_sol.shape[0])
    #pos_sol = [pos_sol[18]]
    full_results = []

    for act in pos_sol:
        # get our KNN
        print(act)

        av_results = dict(metrics=act, av_accuracy=0, av_time=0, accuracy=[], time=[])
        knn = kNNAlgorithm(n_neighbors=int(act[0]), policy=act[2], metric=act[3])

        # preprocess de data folds
        processed_k_x = preprocess_fold(k_x, act[1], int(act[0]), name_db)

        for act_fold in processed_k_x:
            # get data from actual fold
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


def preprocess_fold(folds, weights, n_neigh, name):
    """
    Apply the implemented knn for each possible parameter combination, extract the metrics and store to a csv file.
    :param folds: folds (validation_data, train_data, meta_data) that we are going to preprocess
    :param weights: weights policy that we are going to use.
    :param n_neigh: number of neighbours of our execution
    :param name: string with the name of our dataset.
    """
    if name == "hypothyroid":
        preprocess = preprocess_hypothyroid
    elif name == "grid":
        preprocess = preprocess_grid

    preprocessed_folds = []
    for act in folds:
        (X_train, y_train), (X_val, y_val) = preprocess(act['db_train'], act['db_val'], act['meta'], weights, n_neigh)
        preprocessed_folds.append({
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        })
    return preprocessed_folds


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

