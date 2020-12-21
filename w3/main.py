from evaluation.main_functions import *
from dataPreprocessing.read_data import read_database as read_database


def main():
    """
    choose the dataset to analyze.
     OPTIONS:
        -> "grid"
        -> "hypothyroid"
    """
    database_name = "grid"
    res_db = read_database(database_name)

    # name of the file where are stored the metrics obtained by the execution of each possible combination of input
    # parameters for KNN algorithm.
    n_file_metrics = database_name+"_metrics"

    # name of the file where the Dataframe of the possible combinations are ranked after statistical comparison.
    # Only the combinations that passed threshold of the test.
    n_file_bests = database_name+"_bests_comb"

    # name of the file where are stored the results of the execution of the 4 reduction algorithms that we choose
    # (enn, menn, fcnn and drop3) and the non reduced one for the best combination of KNN parameters extracted before.
    n_file_reduct_metrics = database_name + "_reduct_metrics"

    # name of the file where are stored the results rank after the statistical analysis for the KNN results after
    # applying the reduction algorithms. Only the algorithms that passed the test threshold.
    n_file_reduct_bests = database_name+"_reduct_bests_comb"

    ##############################################################################
    # The following functions CAN BE EXECUTED ISOLATED, commenting the others
    # apply knn to all the possible combination of parameters and store in a csv file
    best_knn_metrics(res_db, n_file_metrics)

    # find the best knn parameter combination and store in a csv file
    best_knn_statistical_comp(n_file_metrics, n_file_bests, reduced=False)

    # get the reduction of the best knn
    reduct_best_knn(n_file_bests, n_file_reduct_metrics, res_db)
    
    # get the rank between the reduced algorithms + non_reduced knn (with the best combination of parameters for knn)
    best_knn_statistical_comp(n_file_reduct_metrics, n_file_reduct_bests, reduced=True)

if __name__ == "__main__":
    main()
