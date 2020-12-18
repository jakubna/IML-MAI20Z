from evaluation.main_functions import *
from dataPreprocessing.read_data import read_and_preprocess_database as read_and_preprocess_database


def main():
    """
    choose the dataset to analyze.
     OPTIONS:
        -> "grid"
        -> "hypothyroid"
    """
    database_name = "grid"
    res_db = read_and_preprocess_database(database_name)

    # name of the file where you want to store and read the metrics obtained
    n_file = "grid_metrics"

    # apply knn to all the possible combination of parameters
    best_knn_metrics(res_db, n_file)

    # find the best knn parameter combination
    best_knn_get_best_comb(n_file)

    # get the reduction of the best knn


if __name__ == "__main__":
    main()

