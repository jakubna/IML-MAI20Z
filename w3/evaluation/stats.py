import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


# analyse two by two configurations using t-test and compare the overall performances of classifiers counting the number of data sets on which an algorithm is the overall winner
# call get_best_results(results, reduced = False)
def compute_stat_test(sample1, sample2):
    """
    Calculate the t-test on TWO RELATED samples of scores.
    param sample1: the metric results of one model configuration (accuracy/time/datareduction)
    param sample1: the metric results of other model configuration (accuracy/time/datareduction)
    return t-statistic and Two-sided p-value.
    """
    try:
        stat, p = ttest_ind(sample1, sample2)
    except ValueError:
        stat, p = 0, 1
    return {'stat': stat, 'p': p}


def eval_stat_test(mat, alpha=0.05):
    """
    Evaluate each pair of configuration to understand which one is better
    param mat: matrix containing all resultes of compute_stat_ind for each configuration model
    return matrix with the indication of the bests configuration of each pair
    """
    best_mat = np.zeros(mat.shape[:-1])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            statistic = mat[i][j][0]
            p_value = mat[i][j][1]
            if p_value < alpha:  # significant
                if statistic < 0:  # second is better
                    best_mat[i][j] = 2
                elif statistic > 0:  # first is better
                    best_mat[i][j] = 1
                else:  # tie
                    best_mat[i][j] = 0
            else:  # tie
                best_mat[i][j] = 0
    return best_mat


def stats_mat(results, metric):
    """
    compute the matrix containing all results of compute_stat_ind for each configuration model
    param results: structure with all accuracy and time obtained for each run (10) of each configuration model
    return one matrix containing all results of compute_stat_ind for each configuration model
    """
    stats = np.full(shape=(len(list(results['model'].unique())), len(list(results['model'].unique())), 2), fill_value=np.nan)

    # considering results as a dataframe with columns= ['model', 'accuracy', 'time']

    for i, model1 in enumerate(list(results['model'].unique())):
        for j, model2 in enumerate(list(results['model'].unique())):
            if model1 == model2:
                continue

            res1 = list(results[results['model'] == model1][metric])
            res2 = list(results[results['model'] == model2][metric])

            stat = compute_stat_test(res1, res2)

            stats[i, j] = np.array([stat['stat'], stat['p']])

    return stats


def get_best_ind(results, reduced=False):
    """
    Get the ind of the statistical bests configurations
    param results: dataframe with all accuracy and time obtained for each run (10) of each configuration model
    param reduced: if the data was reduced and the results has a column 'storage' it should be set to True
    return a list containing the ind of the statistical best configurations
    """

    stats_accuracy = stats_mat(results, 'accuracy')
    stats_time = stats_mat(results, 'time')
    best_mat_acc = eval_stat_test(stats_accuracy)
    best_mat_time = eval_stat_test(stats_time)

    if reduced:
        stats_storage = stats_mat(results, 'storage')
        best_mat_storage = eval_stat_test(stats_storage)

    N = best_mat_acc.shape[0]
    threshold = N / 2 + 1.96 * (N ** 0.5) / 2  # use of z-test: if the number of wins is at least N/2 + 1.96(N^0.5)/2,
                                            # the algorithm is significantly better with p < 0.05

    best_accuracy_idx = str(np.argwhere(np.sum(best_mat_acc < 2, axis=1) >= threshold).flatten())
    best_accuracy_idx = best_accuracy_idx.replace('[', '')
    best_accuracy_idx = best_accuracy_idx.replace(']', '')
    best_accuracy_idx = best_accuracy_idx.split()

    best_time_idx = str(np.argwhere(np.sum(best_mat_time == 1, axis=1) >= threshold).flatten())
    best_time_idx = best_time_idx.replace('[', '')
    best_time_idx = best_time_idx.replace(']', '')
    best_time_idx = best_time_idx.split()

    intersection_ = list(set(best_accuracy_idx).intersection(set(best_time_idx)))
    intersection_ = (sorted(list(np.int_(intersection_))))

    if reduced:
        best_storage_idx = str(np.argwhere(np.sum(best_mat_storage < 2, axis=1) >= threshold).flatten())
        best_storage_idx = best_storage_idx.replace('[', '')
        best_storage_idx = best_storage_idx.replace(']', '')
        best_storage_idx = best_storage_idx.split()
        intersection_ = list(set(intersection_).intersection(set(best_storage_idx)))
        intersection_ = (sorted(list(np.int_(intersection_))))

    return intersection_


def get_best_results(results, reduced=False):
    """
    Get the results of accuracy, time and storage (if reduced = True) of the statistical bests configurations
    param results: array of dictionaries from a csv file with all accuracy and time obtained for each run (10) of each configuration model
    param reduced: if the data was reduced and the results has a column 'storage' it should be set to True
    return a dataframe containing the average accuracy and time of the statistical best configurations sorted by the KPI = accuracy/time
    """

    accuracy = []
    time = []
    model_list = []
    for i, model1 in enumerate(results):
        res1_acc = model1['accuracy']
        res1_acc=list(np.fromstring(res1_acc[1:-1], dtype=np.float, sep=' '))
        res1_time = model1['time']
        res1_time=list(np.fromstring(res1_time[1:-1], dtype=np.float, sep=' '))
        metrics=model1['metrics']
        metrics=metrics.strip("']['").split(' ')
        res1_k = metrics[0]
        res1_w = metrics[1]
        res1_v = metrics[2]
        res1_d = metrics[3]
        model = [str(res1_k) + "-" + str(res1_w) + '-' + str(res1_v) + '-' + str(res1_d)] * 10
        accuracy = accuracy + res1_acc
        time = time + res1_time
        model_list = model_list + model

    df_results = pd.DataFrame()
    df_results['model'] = model_list
    df_results['accuracy'] = accuracy
    df_results['time'] = time

    best_models = get_best_ind(df_results, reduced)
    best_results = df_results.groupby(['model']).mean().reset_index().iloc[best_models, :]
    best_results['accuracy/time'] = best_results['accuracy'] / best_results['time']
    best_results = best_results.sort_values(['accuracy/time'], ascending=False)
    best_results=best_results.reset_index()
    best_results=best_results.drop(['index'],axis=1)

    #print(best_results)

    # theoretically, the best configuration is the first row

    return best_results
