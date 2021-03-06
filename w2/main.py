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
        k: number of clusters for KMeans, in this Work 2 this parameter is fix. (NON MODIFICABLE)
        db_name: the used data set name (NON MODIFICABLE)
        max_it: maximum number of iterations.
        seed: seed for the random search of first centroids.
        tol: algorithms tolerance to movement of centroids between loops.
        n_components: number of components that finally you want to obtain after the pca reduction or 
                        sum of the variance ratio to use as a threshold to determine the number of components
                        if null (or None), number of components=number of features.        
    """
    parameters = dict(max_it=100, seed=1, tol=1e-5, n_components=0.90,
                      k=len(set(res['label_true'])), db_name=database_name)

    # make sure that the names of the rows are column names of the database and n_components=length of names_row
    components = get_features(res['data_frame'], parameters['n_components'])

    # apply all the algorithms
    apply_algorithms(res['db'], res['label_true'], parameters, components, database_name)


if __name__ == "__main__":
    main()
