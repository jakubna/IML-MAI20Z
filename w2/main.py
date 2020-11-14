from w2.evaluation.main_functions import *


def main():
    """
    choose the dataset to analyze.
     OPTIONS:
        -> "breast"
        -> "cmc"
        -> "adult"  (warning huge dataset, it can take some minutes to run all the algorithm)
    """
    database_name = "breast"
    res = preprocess_database(database_name)

    """
    define the input parameters for the algorithms:
        k: number of clusters for KMeans, KMedian, BisectingKMeans and FuzzyCMeans algorithms.
        max_it: maximum number of iterations.
        seed: seed for the random search of first centroids.
        tol: algorithms tolerance to movement of centroids between loops.
        n_components: number of components that finally you want to obtain after the pca reduction or 
                        sum of the variance ratio to use as a threshold to determine the number of components
                        if null, number of componentes=number of features        
        db_name: NON changeable item, is the used data set name
    """
    parameters = dict(k=len(set(res['label_true'])), max_it=100, seed=1, tol=1e-5, n_components=0.9, db_name=database_name)

    # make sure that the names of the rows are column names of the database and n_components=length of names_row
    components = get_features(res['data_frame'])

    # apply all the algorithms
    apply_algorithms(res['db'], res['label_true'], parameters, components, database_name)


if __name__ == "__main__":
    main()
