from evaluation.main_functions import *
from dataPreprocessing.read_data import read_database as read_database


def main():
    """
    choose the dataset to analyze.
     OPTIONS:
        -> "grid"
        -> "hypothyroid"
    """
    database_name = "hypothyroid"
    res_db = read_database(database_name)

    # name of the file where you want to store and read the metrics obtained
    n_file_metrics = database_name+"_metrics"
    n_file_bests = database_name+"_bests_comb"

    # apply knn to all the possible combination of parameters
    # best_knn_metrics(res_db, database_name, database_name)

    # find the best knn parameter combination
    # best_knn_get_best_comb(n_file_metrics, n_file_bests)

    # get the reduction of the best knn
    reduct_best_knn(n_file_bests, res_db, database_name)


if __name__ == "__main__":
    main()
