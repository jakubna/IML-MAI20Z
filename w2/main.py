from w2.evaluation.main_functions import *


def main():
    """
    choose the dataset to analyze.
     OPTIONS:
        -> "breast"
        -> "cmc"
        -> "adult"  (warning huge dataset, it can take some minutes to run all the algorithm)
    """
    database_name = "cmc"
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
    parameters = dict(k=3, max_it=100, seed=1, tol=1e-5, n_components=3, db_name=database_name)

    # apply all the algorithms
    apply_algorithms(res['db'], res['label_true'], parameters)

    # load results to a csv file
    # set_output(result, database_name)


if __name__ == "__main__":
    main()
