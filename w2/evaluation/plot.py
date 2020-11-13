import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot3d(data_input, y, titles):
    """ the function aims to reduce the features to two dimensions using PCA method and plot the clusters
        data_input: 3D data array of size (rows, features).
        y: results labels.
        title: graphic title (example: adults dataset using KMeans).
    """
    fig = plt.figure(figsize=(10, 10))
    for i in range(0, len(titles)):
        y_predict = y[i]
        pca = PCA(n_components=3)
        tsne = TSNE(n_components=3)
        principal_components_pca = pca.fit_transform(data_input)
        principal_components_tsne = tsne.fit_transform(data_input)

        final_df_pca = pd.DataFrame(data=principal_components_pca,
                                    columns=['principal component 1', 'principal component 2', 'principal component 3'])
        final_df_tsne = pd.DataFrame(data=principal_components_tsne,
                                     columns=['principal component 1', 'principal component 2',
                                              'principal component 3'])

        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.set_xlabel('Principal Component 1', fontsize=10)
        ax.set_ylabel('Principal Component 2', fontsize=10)
        ax.set_zlabel('Principal Component 3', fontsize=10)
        ax.set_title('3 component PCA: ' + titles[i], fontsize=15)
        targets = list(set(y_predict))
        y_predict = pd.DataFrame(y_predict).iloc[:, -1]
        colors = ['r', 'g', 'k', 'y', 'c', 'm']
        for target, color in zip(targets, colors):
            indices2keep = y_predict == target
            ax.scatter(final_df_pca.loc[indices2keep, 'principal component 1'],
                       final_df_pca.loc[indices2keep, 'principal component 2'],
                       final_df_pca.loc[indices2keep, 'principal component 3'],
                       c=color,
                       s=50)
        ax.legend(targets)
        ax.grid()

        ax = fig.add_subplot(2, 2, i + 1 + (len(titles)), projection='3d')
        ax.set_xlabel('Principal Component 1', fontsize=10)
        ax.set_ylabel('Principal Component 2', fontsize=10)
        ax.set_zlabel('Principal Component 3', fontsize=10)
        ax.set_title('3 component TSNE: ' + titles[i], fontsize=15)
        targets = list(set(y_predict))
        y_predict = pd.DataFrame(y_predict).iloc[:, -1]
        colors = ['r', 'g', 'k', 'y', 'c', 'm']
        for target, color in zip(targets, colors):
            indices2keep = y_predict == target
            ax.scatter(final_df_tsne.loc[indices2keep, 'principal component 1'],
                       final_df_tsne.loc[indices2keep, 'principal component 2'],
                       final_df_tsne.loc[indices2keep, 'principal component 3'],
                       c=color,
                       s=50)
        ax.legend(targets)
        ax.grid()
    plt.show()


def plot2d(data_input, y, titles):
    """ the function aims to reduce the features to two dimensions using PCA method and plot the clusters
        data_input: 3D data array of size (rows, features).
        y: results labels.
        title: graphic title (example: adults dataset using KMeans).
    """
    fig = plt.figure(figsize=(10, 10))
    for i in range(0, len(titles)):
        y_predict = y[i]
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2)
        principal_components_pca = pca.fit_transform(data_input)
        principal_components_tsne = tsne.fit_transform(data_input)

        final_df_pca = pd.DataFrame(data=principal_components_pca,
                                    columns=['principal component 1', 'principal component 2'])
        final_df_tsne = pd.DataFrame(data=principal_components_tsne,
                                     columns=['principal component 1', 'principal component 2'])
        ax = fig.add_subplot(2, 2, i + 1)
        ax.set_xlabel('Principal Component 1', fontsize=10)
        ax.set_ylabel('Principal Component 2', fontsize=10)
        ax.set_title('2 component PCA: ' + titles[i], fontsize=15)
        targets = list(set(y_predict))
        y_predict = pd.DataFrame(y_predict).iloc[:, -1]
        colors = ['r', 'g', 'k', 'y', 'c', 'm']
        for target, color in zip(targets, colors):
            indices2keep = y_predict == target
            ax.scatter(final_df_pca.loc[indices2keep, 'principal component 1'],
                       final_df_pca.loc[indices2keep, 'principal component 2'],
                       c=color,
                       s=50)
        ax.legend(targets)
        ax.grid()

        ax = fig.add_subplot(2, 2, i + 1 + (len(titles)))
        ax.set_xlabel('Principal Component 1', fontsize=10)
        ax.set_ylabel('Principal Component 2', fontsize=10)
        ax.set_title('2 component TSNE: ' + titles[i], fontsize=15)
        targets = list(set(y_predict))
        y_predict = pd.DataFrame(y_predict).iloc[:, -1]
        colors = ['r', 'g', 'k', 'y', 'c', 'm']
        for target, color in zip(targets, colors):
            indices2keep = y_predict == target
            ax.scatter(final_df_tsne.loc[indices2keep, 'principal component 1'],
                       final_df_tsne.loc[indices2keep, 'principal component 2'],
                       c=color,
                       s=50)
        ax.legend(targets)
        ax.grid()
    plt.show()


def plot_ori_3d(x, x_pca, true_labels, names):
    """
     Method that plot function without pca or t-nse reduction in 3D.
     :param x: processed dataset 2D numpy array.
     :param x_pca: processed data set with a pca reduction, 2d numpy array.
     :param true_labels: labels of the real classification extracted from the database.
     :param names: names of the features chosen by user to plot for the original data set plot.
     """
    fig = plt.figure(figsize=(10, 10))
    final_df = pd.DataFrame(data=x, columns=[names[0], names[1], names[2]])
    final_df_pca = pd.DataFrame(data=x_pca,
                                columns=['principal component 1', 'principal component 2', 'principal component 3'])

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlabel(names[0], fontsize=10)
    ax.set_ylabel(names[1], fontsize=10)
    ax.set_zlabel(names[2], fontsize=10)
    ax.set_title('Original: ', fontsize=15)
    targets = list(set(true_labels))
    y_predict = pd.DataFrame(true_labels).iloc[:, -1]
    colors = ['r', 'g', 'k', 'y', 'c', 'm']
    for target, color in zip(targets, colors):
        indices2keep = y_predict == target
        ax.scatter(final_df.loc[indices2keep, names[0]],
                   final_df.loc[indices2keep, names[1]],
                   final_df.loc[indices2keep, names[2]],
                   c=color,
                   s=50)
    ax.legend(targets)
    ax.grid()

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=10)
    ax.set_ylabel('Principal Component 2', fontsize=10)
    ax.set_zlabel('Principal Component 3', fontsize=10)
    ax.set_title('Our 3 component PCA: ', fontsize=15)
    targets = list(set(y_predict))
    y_predict = pd.DataFrame(y_predict).iloc[:, -1]
    colors = ['r', 'g', 'k', 'y', 'c', 'm']
    for target, color in zip(targets, colors):
        indices2keep = y_predict == target
        ax.scatter(final_df_pca.loc[indices2keep, 'principal component 1'],
                   final_df_pca.loc[indices2keep, 'principal component 2'],
                   final_df_pca.loc[indices2keep, 'principal component 3'],
                   c=color,
                   s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


def plot_ori_2d(x, x_pca, true_labels, names):
    """
    Method that plot function without pca or t-nse reduction in 2D.
    :param x: processed dataset 2D numpy array.
    :param x_pca: processed data set with a pca reduction, 2d numpy array.
    :param true_labels: labels of the real classification extracted from the database.
    :param names: names of the features chosen by user to plot for the original data set plot.
    """
    fig = plt.figure(figsize=(10, 10))
    final_df = pd.DataFrame(data=x, columns=[names[0], names[1]])
    final_df_pca = pd.DataFrame(data=x_pca, columns=['principal component 1', 'principal component 2'])

    ax = fig.add_subplot(1, 2, 1)
    ax.set_xlabel(names[0], fontsize=10)
    ax.set_ylabel(names[1], fontsize=10)
    ax.set_title('Original: ', fontsize=15)
    targets = list(set(true_labels))
    y_predict = pd.DataFrame(true_labels).iloc[:, -1]
    colors = ['r', 'g', 'k', 'y', 'c', 'm']
    for target, color in zip(targets, colors):
        indices2keep = y_predict == target
        ax.scatter(final_df.loc[indices2keep, names[0]],
                   final_df.loc[indices2keep, names[1]],
                   c=color,
                   s=50)
    ax.legend(targets)
    ax.grid()

    ax = fig.add_subplot(1, 2, 2)
    ax.set_xlabel('Principal Component 1', fontsize=10)
    ax.set_ylabel('Principal Component 2', fontsize=10)
    ax.set_title('Our 2 component PCA: ', fontsize=15)
    targets = list(set(y_predict))
    y_predict = pd.DataFrame(y_predict).iloc[:, -1]
    colors = ['r', 'g', 'k', 'y', 'c', 'm']
    for target, color in zip(targets, colors):
        indices2keep = y_predict == target
        ax.scatter(final_df_pca.loc[indices2keep, 'principal component 1'],
                   final_df_pca.loc[indices2keep, 'principal component 2'],
                   c=color,
                   s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

