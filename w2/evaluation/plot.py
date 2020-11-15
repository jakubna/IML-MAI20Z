import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot3d(data_input, y, titles, components_names, reduct=None):
    """
    the function aims to reduce the features to two dimensions using PCA method and plot the clusters
    :param data_input: 3D data array of size (rows, features).
    :param y: results labels.
    :param titles: graphic title (example: adults dataset using KMeans).
    :param components_names: array with the component names for each plot.
    :param reduct: array with the same dimension of data_input, it has if a reduction of data
                    is needed and the reduction type.
    """
    fig = plt.figure(figsize=(10, 10))
    for i in range(0, len(titles)):
        name_compos = components_names[i]
        y_predict = y[i]
        principal_components = data_input[i][:, :3]
        if reduct[i] is not None:
            principal_components = pca_tnse_reduction(data_input[i], 3, reduct[i])

        final_df = pd.DataFrame(data=principal_components,
                                columns=['component 1', 'component 2', 'component 3'])

        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.set_xlabel(name_compos[0], fontsize=10)
        ax.set_ylabel(name_compos[1], fontsize=10)
        ax.set_zlabel(name_compos[2], fontsize=10)
        ax.set_title('3D ' + titles[i], fontsize=15)
        targets = list(set(y_predict))
        y_predict = pd.DataFrame(y_predict).iloc[:, -1]
        colors = ['r', 'g', 'k', 'y', 'c', 'm']
        for target, color in zip(targets, colors):
            indices2keep = y_predict == target
            ax.scatter(final_df.loc[indices2keep, 'component 1'],
                       final_df.loc[indices2keep, 'component 2'],
                       final_df.loc[indices2keep, 'component 3'],
                       c=color,
                       s=50)
        ax.legend(targets)
        ax.grid()
    plt.show()


def plot2d(data_input, y, titles, components_names, reduct=None):
    """
    the function aims to reduce the features to two dimensions using PCA method and plot the clusters
    :param data_input: 3D data array of size (rows, features).
    :param y: results labels.
    :param titles: graphic title (example: adults dataset using KMeans).
    :param components_names: array with the component names for each plot.
    :param reduct: array with the same dimension of data_input, it has if a reduction
            of data is needed and the reduction type.
    """
    fig = plt.figure(figsize=(10, 10))
    for i in range(0, len(titles)):
        name_compos = components_names[i]
        y_predict = y[i]
        principal_components = data_input[i][:, :2]
        if reduct[i] is not None:
            principal_components = pca_tnse_reduction(data_input[i], 2, reduct[i])

        final_df = pd.DataFrame(data=principal_components,
                                columns=['component 1', 'component 2'])
        ax = fig.add_subplot(2, 3, i + 1)
        ax.set_xlabel(name_compos[0], fontsize=10)
        ax.set_ylabel(name_compos[1], fontsize=10)
        ax.set_title('2D ' + titles[i], fontsize=15)
        targets = list(set(y_predict))
        y_predict = pd.DataFrame(y_predict).iloc[:, -1]
        colors = ['r', 'g', 'k', 'y', 'c', 'm']
        for target, color in zip(targets, colors):
            indices2keep = y_predict == target
            ax.scatter(final_df.loc[indices2keep, 'component 1'],
                       final_df.loc[indices2keep, 'component 2'],
                       c=color,
                       s=50)
        ax.legend(targets)
        ax.grid()
    plt.show()


def pca_tnse_reduction(data, n_comp, reduct):
    """
    Apply the reduction needed and return the results obtained.
    :param data: 2D data array of size (rows, features).
    :param n_comp: the number ob components that database have to be reduced.
    :param reduct: type of reduction needed.
    :return principal components obtained.
    """
    principal_components = None
    if reduct == 'pca':
        pca = PCA(n_components=n_comp)
        principal_components = pca.fit_transform(data)
    elif reduct == 'tsne':
        tsne = TSNE(n_components=n_comp)
        principal_components = tsne.fit_transform(data)

    return principal_components
