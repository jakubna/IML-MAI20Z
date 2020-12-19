from evaluation.plot import *
from evaluation.evaluate import *
from algorithms.reduction.drop3 import drop3_reduction
from algorithms.reduction.enn import enn_reduction
from algorithms.reduction.fcnn import fcnn_reduction
from algorithms.reduction.menn import menn_reduction
from evaluation.stats import get_best_results
import pandas as pd
from algorithms.kNNAlgorithm import kNNAlgorithm
from dataPreprocessing.read_data import preprocess_data
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
    # pos_sol = [pos_sol[18]]
    full_results = []

    for act in pos_sol:
        # get our KNN
        print(act)

        av_results = dict(metrics=act, av_accuracy=0, av_time=0, accuracy=[], time=[])
        knn = kNNAlgorithm(n_neighbors=int(act[0]), policy=act[2], metric=act[3])

        # preprocess de data folds
        processed_k_x = preprocess_data(k_x, act[1], int(act[0]), name_db)

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


def best_knn_get_best_comb(name_file_input, name_file_output):
    """
    Apply the evaluation of the metrics extracted for each combination of parameters.
    :param name_file_input: the name of the file on where you want to read the metrics.
    :param name_file_output: the name of the file on where you want to write the best combinations obtained.
    """
    # read the combination metrics
    metrics = read_csv(name_file_input)

    # get the bests combinations
    bests = get_best_results(metrics)

    # store the best combinations
    set_output(bests, name_file_output)


def reduct_best_knn(name_file_input, k_x, name_db):
    """
    Apply the reduction and make the analysis of the metrics extracted.
    :param name_file_input: the name of the file on where you want to read the metrics.
    :param k_x: data array of size (n_folds), each element is a dictionary with validation_data, train_data, meta_data.
    :param name_db: string with the name of our dataset.
    """
    cases = ['full', 'enn', 'menn', 'fcnn', 'drop3']
    # read the best combinations
    best_list = read_csv(name_file_input, metrics=False)

    best = best_list.loc[[0]].to_dict(orient='records')[0]
    comb = best['model'].split('-')

    m = dict(metrics=comb, reduct='', av_accuracy=0, av_time=0, accuracy=[], time=[], storage=0, av_storage=[])
    av_results = np.full(len(cases), m)

    knn = kNNAlgorithm(n_neighbors=int(comb[0]), policy=comb[2], metric=comb[3])

    # preprocess de data folds
    processed_k_x = preprocess_data(k_x, comb[1], int(comb[0]), name_db)

    for i, case in enumerate(cases):
        av_results[i]['reduct'] = case
        for act_fold in processed_k_x:
            # get data from actual fold
            x_validate = act_fold['X_val']
            y_validate = act_fold['y_val']
            # apply or not the reduction
            x_train, y_train = get_reduct(case, act_fold['X_train'], act_fold['y_train'], knn)
            # calculate the average of the storage
            av_results[i]['av_storage'].append(100*len(x_train)/len(act_fold['X_train']))
            t0 = time.time()
            knn.fit(x_train, y_train)
            predict = knn.predict(x_validate)
            t1 = time.time() - t0

            # evaluate the knn
            supervised = evaluate_accuracy(y_validate, predict)

            # sum the new metrics obtained to make average
            av_results[i]['accuracy'].append(supervised['accuracy'])
            av_results[i]['time'].append(t1)

        av_results[i]['accuracy'] = np.array(av_results[i]['accuracy'])
        av_results[i]['time'] = np.array(av_results[i]['time'])
        av_results[i]['av_accuracy'] = np.average(av_results[i]['accuracy'])
        av_results[i]['av_time'] = np.average(av_results[i]['time'])
        av_results[i]['storage'] = np.array(av_results[i]['storage'])
        av_results[i]['av_storage'] = np.average(av_results[i]['storage'])

    print(av_results)


def get_reduct(policy: str, x: np.ndarray, y: np.array, knn: kNNAlgorithm):
    """
    Apply the reduction to data passed by parameter.
    :param policy: algorithm chosen to make the reduction.
    :param x: data ndarray with all the samples of the training part of the fold.
    :param y: labels for each sample described in x.
    :param knn: Knn algorithm that is set with parameters previously.
    """
    if policy == 'full':
        return x, y
    elif policy == 'enn':
        return enn_reduction(knn, x, y)
    elif policy == 'menn':
        return menn_reduction(knn, x, y)
    elif policy == 'fcnn':
        return fcnn_reduction(knn, x, y)
    elif policy == 'drop3':
        return drop3_reduction(knn, x, y)
    else:
        raise ValueError('The reduction algorithm chosen is supported')


def set_output(results, database_name):
    # print result at the terminal
    print(results)

    # load results to a csv file
    results.to_csv("./results/" + database_name + ".csv", sep='\t', encoding='utf-8', index=False)

    print("\nThe CSV output files are created in results folder of this project\n")


def read_csv(name_file, metrics=True):
    """
    Read the csv file and extract the metrics for each combination.
    :param name_file: the name of the file on where you want to read the metrics.
    :param metrics: boolean that change the mode read metrics file or read bests file
    """
    results = pd.read_csv("./results/" + name_file + ".csv", sep='\t', encoding='utf-8')

    if metrics:
        res = np.array(results.to_dict(orient='records'))
        for act in res:
            act['time'] = list(np.fromstring(act['time'][1:-1], dtype=np.float, sep=' '))
            act['accuracy'] = list(np.fromstring(act['accuracy'][1:-1], dtype=np.float, sep=' '))
            act['metrics'] = act['metrics'].strip("][").split(' ')
    else:
        res = results

    return res
