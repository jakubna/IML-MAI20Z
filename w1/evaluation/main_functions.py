from w1.dataPreprocessing.breast import preprocess as preprocess_breast
from w1.dataPreprocessing.cmc import preprocess as preprocess_cmc
from w1.dataPreprocessing.adult import preprocess as preprocess_adult
from w1.algorithms.KMeans import KMeans
from w1.algorithms.bisecting_kmeans import BisectingKMeans
from w1.algorithms.KMedians import KMedians
from w1.algorithms.FuzzyCMeans import FuzzyCMeans
from w1.algorithms.dbscan import *
from w1.evaluation.evaluate import *
from w1.evaluation.plot2D import *
from w1.evaluation.optimize import *
import matplotlib.pyplot as plt


def apply_algorithms(x: np.ndarray, label_true, df, params):
    """
    Apply the implemented algorithms, dbscan and evaluate the obtained results.
    :param x: 2D data array of size (rows, features).
    :param label_true: labels of the real classification extracted from the database.
    :param df: dataframe of the dataframe, needed or dbscan algorithm.
    :param params: dictionary with all the parameters required to execute the algorithms.
    :return: two dataframes with the evaluations results, one for algorithms implemented in this practise another
    for dbscan.
    """
    names = ['KMeans', 'Bisecting KMeans', 'KMedians', 'FuzzyCMeans']
    labels = []

    # KMeans
    algorithm = KMeans(k=params['k'], seed=params['seed'], max_it=params['max_it'], tol=params['tol'])
    labels_kmeans = algorithm.fit_predict(x)
    labels.append(labels_kmeans)

    # Bisecting
    algorithm = BisectingKMeans(k=params['k'], seed=params['seed'], max_it=params['max_it'], tol=params['tol'])
    labels_bisecting = algorithm.fit_predict(x)
    labels.append(labels_bisecting)

    # KMeadians
    algorithm = KMedians(k=params['k'], seed=params['seed'], max_it=params['max_it'], tol=params['tol'])
    labels_kmedians = algorithm.fit_predict(x)
    labels.append(labels_kmedians)

    # FuzzyCMeans
    algorithm = FuzzyCMeans(k=params['k'], seed=params['seed'], max_it=params['max_it'], tol=params['tol'],
                            epsilon=params['epsilon'], m=params['m'])
    fuzzy_values = algorithm.fit_predict(x)
    labels.append(fuzzy_values[2])
    fuzzy_values = fuzzy_values[:2]
    # fuzzy_values -> [memb_matrix, centroids, crisp_labels]

    # DBscan
    # find_eps(x)
    dbscan_results, df = dbscan_(x, df=df, eps=params['eps'])
    # dbscan_results = pd.DataFrame()

    # apply the evaluations of the obtained results
    df_home = apply_evaluation(x, label_true, labels, names, fuzzy_values)

    plot2d(x, labels, names)
    return dict(our_df=df_home, dbscan_df=dbscan_results)


def apply_evaluation(x: np.ndarray, label_true, labels, names, fuzzy_scores):
    """
    Apply all the evaluations to the implemented algorithms and classify in a dataframe.
    :param x: 2D data array of size (rows, features).
    :param label_true: labels of the real classification extracted from the database.
    :param labels: list of the different classifications obtained for all the algorithms.
    :param names: list of all the implemented algorithm names.
    :param fuzzy_scores: fuzzy features extracted from FuzzyCMeans algorithm.
    :return: a dataframe with the evaluation results for algorithms implemented in this practise.
    """
    nan = np.nan
    rows = []

    for i in range(0, len(names)):
        act_name = names[i]
        fuzzy_results = dict(normalized_partition_coefficient=nan, partition_entropy=nan, xie_beni=nan)
        if act_name == "FuzzyCMeans":
            fuzzy_results = evaluate_soft_partitions_internal(x, fuzzy_scores[0], fuzzy_scores[1])
        unsupervised = evaluate_unsupervised_internal(x, labels[i])
        supervised = evaluate_supervised_external(label_true, labels[i])

        row = {**dict(Names=act_name), **supervised, **unsupervised, **fuzzy_results}
        rows.append(row)
    df_results = pd.DataFrame(rows)

    return df_results


def preprocess_database(database: str):
    """
    With the input string choose the dataset that we want to execute and call preprocess function.
    :param database: string with the name of the dataset that we want to execute.
    :return: features of the preprocessed database(processed database, true classification results, complete dataframe).
    """
    # processed -> db, label_true, data_frame
    if database == "breast":
        processed = preprocess_breast()
    elif database == "cmc":
        processed = preprocess_cmc()
    elif database == "adult":
        processed = preprocess_adult()
    else:
        raise ValueError('database not found')

    return processed


def optim_k_value(processed, name: str, params):
    fig = plt.figure(figsize=(20, 20))
    algorithms = [KMeans, KMedians, BisectingKMeans, FuzzyCMeans]
    names = ["KMeans", "KMedians", "BisectingKMeans", "FuzzyCMeans"]

    print("Get all the optimal K score for each algorithm for the dataset: " + name)
    for i in range(0, len(algorithms)):
        test = optimize(x=processed['db'], y=processed['label_true'], algorithm=algorithms[i], metric=params['metric'],
                        k_values=params['rang'], goal=params['goal'], seed=params['seed'])
        print(names[i], test['optimal'][0])
        ax = fig.add_subplot(2, 2, i+1)
        ax.plot(params['rang'], test['plot'], '-ob')
        ax.scatter(test['optimal'][0]['k'], test['optimal'][0]['score'], c='red', alpha=0.5, s=150)
        ax.set_xlabel('k')
        ax.set_ylabel('Method score')
        ax.set_title('The ' + params['metric'] + ' result for '+names[i])

    plt.show()


def set_output(result, database_name):
    # print result at the terminal
    print(result['our_df'])
    print(result['dbscan_df'])

    # load results to a csv file
    result['our_df'].to_csv("./results/"+database_name+"_algorithms_results", sep='\t', encoding='utf-8', index=False)
    result['dbscan_df'].to_csv("./results/"+database_name+"_dbscan_results", sep='\t', encoding='utf-8', index=False)
    print("\nThe CSV output files are created in results folder of this project\n")
