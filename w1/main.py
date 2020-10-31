from w1.evaluation.main_functions import *


def main():
    # choose the dataset to analyze
    database_name = "cmc"
    res = preprocess_database(database_name)

    # define the parameters
    parameters = dict(k=2, max_it=100, seed=1, tol=1e-5, eps=0.5, epsilon=0.01, m=2, optimal_k=False)

    # apply all the algorithms
    result = apply_algorithms(res['db'], res['label_true'], res['data_frame'], parameters)

    # print result at the terminal
    print(result['our_df'])
    print(result['dbscan_df'])

    # load results to a csv file
    result['our_df'].to_csv("results/"+database_name+"_algorithms_results", sep='\t', encoding='utf-8', index=False)
    result['dbscan_df'].to_csv("results/"+database_name+"_dbscan_results", sep='\t', encoding='utf-8', index=False)
    print("\nThe CSV output files are created in results folder of this project\n")

    # find optimal K value for each algorithm
    optim_k_value(res, database_name, parameters['seed'])


if __name__ == "__main__":
    main()


