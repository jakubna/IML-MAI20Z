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
    :param components: name and index of the features chosen by user to plot for the original data set plot.
    """
    names = ['Original dataset', 'Our PCA results', 'KMeans with previous our PCA reduction',
             'KMeans without previous reduction (PCA)', 'KMeans without previous reduction (T-SNE)']

    datasets = []
    labels = []
    reduct = []

    # get the representation of the original matrix splitted to be plotted
    partial_x = split_db_original(x, components)
    datasets.append(partial_x)
    labels.append(label_true)
    reduct.append(None)

    # get our PCA
    pca = PCA(n_components=params['n_components'])
    our_pca = pca.fit_transform(x)
    datasets.append(our_pca)
    labels.append(label_true)
    reduct.append(None)

    # get PCA and IPCA from sklearn
    sk_pca = pca_sklearn(x, params['db_name'], params['n_components'])
    sk_ipca = ipca_sklearn(x, params['db_name'], params['n_components'])

    # compare the three PCA algorithms
    name = ['Our PCA', 'SK_PCA', 'SK_IPCA']
    pca_data = [our_pca, sk_pca['db'], sk_ipca['db']]
    apply_evaluation(pca_data, label_true, params, name)

    # KMeans with PCA reduction
    algorithm = KMeans(k=params['k'], seed=params['seed'], max_it=params['max_it'], tol=params['tol'])
    labels_kmeans = algorithm.fit_predict(our_pca)
    datasets.append(our_pca)
    labels.append(labels_kmeans)
    reduct.append(None)

    # KMeans without PCA reduction
    algorithm = KMeans(k=params['k'], seed=params['seed'], max_it=params['max_it'], tol=params['tol'])
    labels_kmeans = algorithm.fit_predict(x)
    datasets.append(x)
    labels.append(labels_kmeans)
    reduct.append('pca')
    datasets.append(x)
    labels.append(labels_kmeans)
    reduct.append('tsne')

    if params['n_components'] == 2:
        pca_names = ['PCA Component 1', 'PCA Component 2']
        plot_names = [components[0], pca_names, pca_names, pca_names, ['TSNE 1', 'T-SNE 2']]
        plot2d(datasets, labels, names, plot_names, reduct)
    elif params['n_components'] == 3:
        pca_names = ['PCA Component 1', 'PCA Component 2', 'PCA Component 3']
        plot_names = [components[0], pca_names, pca_names, pca_names, ['TSNE 1', 'T-SNE 2', 'T-SNE 3']]
        plot3d(datasets, labels, names, plot_names, reduct)


def apply_evaluation(x, label_true, params, names):
    """
    Apply all the evaluations to the implemented algorithms and classify in a dataframe.
    :param x: 2D data array of size (rows, features).
    :param label_true: labels of the real classification extracted from the database.
    :param params: parameters for the k-means algorithm.
    :param names: list of all the evaluated algorithms.
    :return: a dataframe with the evaluation results for algorithms implemented in this practise.
    """
    rows = []

    for i in range(0, len(names)):
        act_name = names[i]
        act_data = x[i]

        algorithm = KMeans(k=params['k'], seed=params['seed'], max_it=params['max_it'], tol=params['tol'])
        labels = algorithm.fit_predict(act_data)

        unsupervised = evaluate_unsupervised_internal(act_data, labels)
        supervised = evaluate_supervised_external(label_true, labels)

        row = {**dict(Names=act_name), **supervised, **unsupervised}
        rows.append(row)
    df_results = pd.DataFrame(rows)
    set_output(df_results, 'pca_analysis')


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


def set_output(results, database_name):
    # print result at the terminal
    print(results)

    # load results to a csv file
    results.to_csv("./results/" + database_name + ".csv", sep='\t', encoding='utf-8', index=False)

    print("\nThe CSV output files are created in results folder of this project\n")
