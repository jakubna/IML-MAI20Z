from w2.dataPreprocessing.breast import preprocess as preprocess_breast
from w2.dataPreprocessing.cmc import preprocess as preprocess_cmc
from w2.dataPreprocessing.adult import preprocess as preprocess_adult
from w2.algorithms.KMeans import KMeans
from w2.algorithms.pca import PCA
from w2.evaluation.plot import *
from w2.evaluation.evaluate import *
import pandas as pd
from w2.algorithms.pca_sklearn import *


def apply_algorithms(x: np.ndarray, label_true, params, components):
    """
    Apply the implemented algorithms, dbscan and evaluate the obtained results.
    :param x: 2D data array of size (rows, features).
    :param label_true: labels of the real classification extracted from the database.
    :param params: dictionary with all the parameters required to execute the algorithms.
    :param components: name and index of the features choosed by user to plot for the original data set plot.
    """
    names = ['KMeans without previous PCA reduct', 'KMeans with previous PCA reduct']
    labels = []

    # get our PCA
    pca = PCA(n_components=params['n_components'])
    our_pca = pca.fit_transform(x)

    # get PCA and IPCA from sklearn
    sk_pca = pca_sklearn(x, params['db_name'], params['n_components'])
    sk_ipca = ipca_sklearn(x, params['db_name'], params['n_components'])

    # compare the three PCA algorithms
    compare_sklearn_results(our_pca, sk_pca, sk_ipca)

    # KMeans without PCA reduction
    algorithm = KMeans(k=params['k'], seed=params['seed'], max_it=params['max_it'], tol=params['tol'])
    labels_kmeans = algorithm.fit_predict(x)
    labels.append(labels_kmeans)

    # KMeans with PCA reduction
    algorithm = KMeans(k=params['k'], seed=params['seed'], max_it=params['max_it'], tol=params['tol'])
    labels_kmeans = algorithm.fit_predict(our_pca)
    labels.append(labels_kmeans)

    plot_original(x, our_pca, label_true, components)

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


def plot_original(x, x_pca, true_labels, components):
    """
    Method that process input parameters and call the plot function without pca or t-nse reduction.
    :param x: processed dataset 2D numpy array.
    :param x_pca: processed data set with a pca reduction, 2d numpy array.
    :param true_labels: labels of the real classification extracted from the database.
    :param components: names and index of the features chosen by user to plot for the original data set plot.
    """
    n_compo = len(components[1])
    cm = components[1]
    ap = []
    for itera in cm:
        ap.append(x[:, itera].tolist())
    ap_np = np.transpose(np.array(ap))
    if n_compo == 2:
        plot_ori_2d(ap_np, x_pca, true_labels, components[0])
    elif n_compo == 3:
        plot_ori_3d(ap_np, x_pca, true_labels, components[0])


def get_features(data_frame, n_components):
    """
    Function that ask to the user which features want to see in the plot of the original data set.
    :param data_frame: original dataframe.
    :param n_components: number of component that user want to reduce the dataset.
    :return: the names of the features that user choose and its index in the matrix.
    """
    col = data_frame.columns.tolist()[:-1]
    com = 1
    components = []
    index = []
    for n_iter in range(n_components):
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


def set_output(result, database_name):
    # print result at the terminal
    print(result['our_df'])

    # load results to a csv file
    result['our_df'].to_csv("./results/"+database_name+"_algorithms_results", sep='\t', encoding='utf-8', index=False)
    print("\nThe CSV output files are created in results folder of this project\n")

