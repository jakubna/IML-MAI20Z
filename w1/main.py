from w1.evaluation.main_functions import *


def main():
    # choose the dataset to analyze
    database_name = "cmc"
    res = preprocess_database(database_name)

    # define the parameters
    parameters = dict(k=4, max_it=100, seed=-1, tol=1e-5, eps=0.02, min_s=3, epsilon=0.01, m=2)

    # apply all the algorithms
    result = apply_algorithms(res['db'], res['label_true'], res['data_frame'], parameters)

    # print result at the terminal
    print(result['our_df'])
    print(result['dbscan_df'])

    # load results to a csv file
    result['our_df'].to_csv("results/"+database_name+"_algorithms_results", sep='\t', encoding='utf-8', index=False)
    result['dbscan_df'].to_csv("results/"+database_name+"_dbscan_results", sep='\t', encoding='utf-8', index=False)
    print("The output files are created in results folder of this project")

    # metodes auxiliars, find eps, optimal K, fitxer de optimize


if __name__ == "__main__":
    main()
