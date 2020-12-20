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

    # name of the file where you want to store and read the metrics obtained
    n_file_metrics = database_name+"_metrics"
    # name of the file where the dataframe of the possible combinations are ranked after statistical comparison
    n_file_bests = database_name+"_bests_comb"
    # name of the file where you want to store and read the metrics obtained for the redacted knn.
    n_file_redact_metrics = database_name + "_redact_metrics"
    # name of the file where the dataframe of the redacted knn are ranked after statistical comparison
    m_file_redact_bests = database_name+"redact_bests_comb"

    # The following functions CAN BE EXECUTED ISOLATED, commenting the others
    # apply knn to all the possible combination of parameters and store in a csv file
    best_knn_metrics(res_db, n_file_metrics)

    # find the best knn parameter combination and store in a csv file
    best_knn_statistical_comp(n_file_metrics, n_file_bests, False)

    # get the reduction of the best knn
    redact_best_knn(n_file_bests, n_file_redact_metrics, res_db)
    
    # get the rank between the redacted algorithms + non_redacted knn (with the best combination of parameters for knn)
    best_knn_statistical_comp(n_file_redact_metrics, m_file_redact_bests, True)


if __name__ == "__main__":
    main()
