from w1.evaluation.main_functions import *


def main():
    """
    choose the dataset to analyze.
     OPTIONS:
        -> "breast"
        -> "cmc"
        -> "adult"  (warning huge dataset, it can take some minutes to run all the algorithm)
    """
    database_name = "adult"
    res = preprocess_database(database_name)

    """
    define the input parameters for the algorithms:
        k: number of clusters for KMeans, KMedian, BisectingKMeans and FuzzyCMeans algorithms.
        max_it: maximum number of iterations.
        seed: seed for the random search of first centroids.
        tol: algorithms tolerance to movement of centroids between loops.
        eps: epsilon for the DBScan algorithm.
        epsilon: epsilon for FuzzyCMeans algorithm.
        m: the fuzziness index for FuzzyCMeans.
    """
    parameters = dict(k=4, max_it=100, seed=1, tol=1e-5, eps=0.25, epsilon=0.01, m=2)

    # apply all the algorithms
    result = apply_algorithms(res['db'], res['label_true'], res['data_frame'], parameters)

    # load results to a csv file
    set_output(result, database_name)

    """
    define the input parameters for the optimize function, this function prints the optimal k value for each algorithm 
    using une metric to evaluate this particularity
        metric: that is the metric that you want to use to maximize or minimize to search the optimal K value, 
        the available metrics are: 
            adjusted_mutual_info_score,adjusted_rand_score, completeness_score, fowlkes_mallows_score,
            homogeneity_score,v_measure_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score,
            normalized_partition_coefficient, partition_entropy, xie_beni
        seed: seed for the random search of first centroids. It should be the same as the algorithm execution to extract
        congruent results.
        rang: list of K-values that you want to evaluate.
        goal: depending of the metric, to obtain the optimal value you need to optimize or minimize the metric. 
    """
    # find optimal K value for each algorithm
    opt_param = dict(metric='silhouette_score', seed=parameters['seed'], rang=[2, 3, 4, 5, 6, 7, 8, 9, 10], goal='max')
    optim_k_value(res, database_name, opt_param)


if __name__ == "__main__":
    main()



