from dataPreprocessing.breast import preprocess as preprocess_breast
from dataPreprocessing.cmc import preprocess as preprocess_cmc
from dataPreprocessing.adult import preprocess as preprocess_adult
from evaluation.plot import *
from evaluation.evaluate import *
import pandas as pd
from algorithms.kNNAlgorithm import kNNAlgorithm


def apply_algorithms(x: np.ndarray, label_true, params):
    """
    Apply the implemented algorithms, dbscan and evaluate the obtained results.
    :param x: 2D data array of size (rows, features).
    :param label_true: labels of the real classification extracted from the database.
    :param params: dictionary with all the parameters required to execute the algorithms.
    """

    # get our PCA
    knn = kNNAlgorithm(n_neighbors=params['n_neighbors'], weights=params['weights'], policy=params['policy'],
                       metric=params['metric'])
    #partitionate the database
    siz = int(x.shape[0]/2)
    sp_x1 = x[:siz]
    sp_x2 = x[siz:]
    lt1 = label_true[:siz]
    lt2 = label_true[siz:]

    sp_x1 = x
    sp_x2 = x
    lt1 = label_true
    lt2 = label_true

    knn.fit(sp_x1, lt1)
    predict = knn.predict(sp_x2)

    supervised = evaluate_supervised_external(lt2, predict)
    print(supervised)


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

