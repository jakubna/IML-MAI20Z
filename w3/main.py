from evaluation.main_functions import *


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

    parameters = dict(n_neighbors=5, weights='equal', policy='majority_class', metric='minkowski')

    # make sure that the names of the rows are column names of the database and n_components=length of names_row
    # components = get_features(res['data_frame'], parameters['n_components'])

    # apply all the algorithms
    apply_algorithms(res['db'], res['label_true'], parameters)


if __name__ == "__main__":
    main()
