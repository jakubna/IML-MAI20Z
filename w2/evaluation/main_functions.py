from w2.dataPreprocessing.breast import preprocess as preprocess_breast
from w2.dataPreprocessing.cmc import preprocess as preprocess_cmc
from w2.dataPreprocessing.adult import preprocess as preprocess_adult
from w2.algorithms.KMeans import KMeans
from w2.algorithms.pca import PCA
from w2.evaluation.plot import *
from w2.evaluation.evaluate import *
import pandas as pd
from w2.algorithms.pca_sklearn import *


def apply_algorithms(x: np.ndarray, label_true, params):
    """
    Apply the implemented algorithms, dbscan and evaluate the obtained results.
    :param x: 2D data array of size (rows, features).
    :param label_true: labels of the real classification extracted from the database.
    :param params: dictionary with all the parameters required to execute the algorithms.
    :return: two dataframes with the evaluations results, one for algorithms implemented in this practise another
    for dbscan.
    """
    names = ['Original dataset', 'KMeans without PCA reduct', 'KMeans with PCA reduct']
    labels = []

    # get our PCA
    pca = PCA(n_components=params['n_components'])
    our_pca = pca.fit_transform(x)

    # get PCA and IPCA from sklearn
    sk_pca = pca_sklearn(x, params['db_name'], params['n_components'])
    sk_ipca = ipca_sklearn(x, params['db_name'], params['n_components'])

    # compare the three PCA algorithms
    compare_sklearn_results(our_pca, sk_pca, sk_ipca)

    labels.append(label_true)
    # KMeans without PCA reduction
    algorithm = KMeans(k=params['k'], seed=params['seed'], max_it=params['max_it'], tol=params['tol'])
    labels_kmeans = algorithm.fit_predict(x)
    labels.append(labels_kmeans)

    # KMeans with PCA reduction
    algorithm = KMeans(k=params['k'], seed=params['seed'], max_it=params['max_it'], tol=params['tol'])
    labels_kmeans = algorithm.fit_predict(our_pca)
    labels.append(labels_kmeans)

    if params['n_components'] == 2:
        plot2d(x, labels, names)
    elif params['n_components'] == 3:
        plot3d(x, labels, names)


def apply_evaluation(x: np.ndarray, label_true, labels, names):
    """
    Apply all the evaluations to the implemented algorithms and classify in a dataframe.
    :param x: 2D data array of size (rows, features).
    :param label_true: labels of the real classification extracted from the database.
    :param labels: list of the different classifications obtained for all the algorithms.
    :param names: list of all the implemented algorithm names.
    :return: a dataframe with the evaluation results for algorithms implemented in this practise.
    """
    nan = np.nan
    rows = []

    for i in range(0, len(names)):
        act_name = names[i]

        unsupervised = evaluate_unsupervised_internal(x, labels[i])
        supervised = evaluate_supervised_external(label_true, labels[i])

        row = {**dict(Names=act_name), **supervised, **unsupervised}
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


def set_output(result, database_name):
    # print result at the terminal
    print(result['our_df'])

    # load results to a csv file
    result['our_df'].to_csv("./results/"+database_name+"_algorithms_results", sep='\t', encoding='utf-8', index=False)
    print("\nThe CSV output files are created in results folder of this project\n")
