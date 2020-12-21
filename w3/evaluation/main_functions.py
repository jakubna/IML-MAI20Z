from evaluation.plot import *
from evaluation.evaluate import *
import matplotlib as plt 
from evaluation.stats import get_best_results
import pandas as pd
from algorithms.kNNAlgorithm import kNNAlgorithm
from algorithms.reductionKNNAlgorithm import reduction_KNNAlogrithm
import itertools
import time


def best_knn_metrics(k_x: np.ndarray, name_file):
    """
    Apply the implemented knn for each possible parameter combination, extract the metrics and store to a csv file.
    :param k_x: data array of size (n_folds).
    :param name_file: the name of the file on where you want to write.
    """
    params = [[1, 3, 5, 7], ['equal', 'mutual_info', 'relief'], ['majority_class', 'inverse_distance', 'sheppard_work'],
              ['minkowski', 'euclidean', 'chebyshev']]

    # get all combinations that must be evaluated
    pos_sol = np.array(list(itertools.product(*params)))
    print("Number of possible combinations:", pos_sol.shape[0])
    # pos_sol = [pos_sol[18]]
    full_results = []

    for act in pos_sol:
        # get our KNN
        print(act)

        av_results = dict(metrics=act, av_accuracy=0, av_time=0, accuracy=[], time=[])
        knn = kNNAlgorithm(n_neighbors=int(act[0]), policy=act[2], weights=act[1], metric=act[3])

        for act_fold in k_x:
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


def best_knn_statistical_comp(name_file_input, name_file_output, reduced):
    """
    Apply the evaluation of the metrics extracted for each combination of parameters.
    :param name_file_input: the name of the file on where you want to read the metrics.
    :param name_file_output: the name of the file on where you want to write the best combinations obtained.
    :param redacted: if the statistical comparison is made between redacted data.
    """
    # read the combination metrics
    metrics = read_csv(name_file_input, reduced)

    # get the bests combinations
    bests = get_best_results(metrics, reduced)

    # store the best combinations
    set_output(bests, name_file_output)
    if not reduced:
        plot(name_file_output, reduced)


def reduct_best_knn(name_file_input, name_file_output, k_x):
    """
    Apply the reduction and make the analysis of the metrics extracted.
    :param name_file_input: the name of the file on where you want to read the metrics.
    :param k_x: data array of size (n_folds), each element is a dictionary with validation_data, train_data, meta_data.
    :param name_file_output: name for the file where we are going to store the results
    """
    cases = ['full', 'enn', 'menn', 'fcnn', 'drop3']
    # read the best combinations
    best_list = read_csv(name_file_input, False, metrics=False)

    best = best_list.loc[[0]].to_dict(orient='records')[0]
    comb = best['model'].split('-')

    full_results = []

    config = dict(n_neighbors=int(comb[0]), weights=comb[1], policy=comb[2], metric=comb[3])

    for i, case in enumerate(cases):
        comb2 = comb+[case]
        print(comb2)
        av_results = dict(metrics=comb2, av_accuracy=0, av_time=0, accuracy=[], time=[], storage=[], av_storage=0)
        for act_fold in k_x:
            print(time.ctime(time.time()))
            # get data from actual fold
            x_validate = act_fold['X_val']
            y_validate = act_fold['y_val']
            # apply or not the reduction
            knn = kNNAlgorithm(n_neighbors=int(comb[0]), weights=comb[1], policy=comb[2], metric=comb[3])
            # x_train, y_train = get_reduct(case, act_fold['X_train'], act_fold['y_train'], knn)
            x_train, y_train = reduction_KNNAlogrithm(config, act_fold['X_train'], act_fold['y_train'], case)
            t0 = time.time()
            knn.fit(x_train, y_train)
            predict = knn.predict(x_validate)
            t1 = time.time() - t0

            # evaluate the knn
            supervised = evaluate_accuracy(y_validate, predict)

            # sum the new metrics obtained to make average
            av_results['accuracy'].append(supervised['accuracy'])
            av_results['time'].append(t1)
            # calculate the average of the storage
            av_results['storage'].append(100*len(x_train)/len(act_fold['X_train']))

        av_results['accuracy'] = np.array(av_results['accuracy'])
        av_results['time'] = np.array(av_results['time'])
        av_results['av_accuracy'] = np.average(av_results['accuracy'])
        av_results['av_time'] = np.average(av_results['time'])
        av_results['storage'] = np.array(av_results['storage'])
        av_results['av_storage'] = np.average(av_results['storage'])
        full_results.append(av_results)

    print(full_results)
    df_results = pd.DataFrame(full_results)
    set_output(df_results, name_file_output)


def set_output(results, database_name):
    # print result at the terminal
    print(results)

    # load results to a csv file
    results.to_csv("./results/" + database_name + ".csv", sep='\t', encoding='utf-8', index=False)

    print("\nThe CSV output files are created in results folder of this project\n")


def read_csv(name_file, reduced=False, metrics=True):
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
            if reduced==False:
                act['metrics'] = act['metrics'].strip("][").split(' ')
            else:
                act['storage'] = list(np.fromstring(act['storage'][1:-1], dtype=np.float, sep=' '))
                print(act['metrics'])
                act['metrics'] = act['metrics'].strip("][").split(', ')
    else:
        res = results

    return res


def plot(name_file_input, reduced):
    """
        Plot the results extracted from the input file.
        :param name_file_input: the name of the file on where you want to read the metrics.
        :param reduced: boolean that indicates if there are reduced format
    """
    metrics = read_csv(name_file_input, reduced, metrics=False)
    steps = (int)(metrics.shape[0]/3)
    green = np.array(np.full(steps, 'green'))
    blue = np.array(np.full(steps, 'blue'))
    red = np.array(np.full((metrics.shape[0])-steps*2, 'red'))
    colors = np.concatenate((green, blue, red), axis=None)
    if reduced:
        metrics.drop("accuracy/time", axis=1, inplace=True)
        metrics.drop("model", axis=1, inplace=True)
        metrics.plot(x='accuracy', y='time', z='storage', kind='scatter', c=colors)
        plt.show()
    else:
        metrics.drop("accuracy/time", axis=1, inplace=True)
        metrics.drop("model", axis=1, inplace=True)
        metrics.plot(x='accuracy', y='time', kind='scatter', c=colors)
        plt.show()
