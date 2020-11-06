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

        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
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

        ax = fig.add_subplot(2, 3, i + 1 + (len(titles)), projection='3d')
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
        ax = fig.add_subplot(2, 3, i + 1)
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

        ax = fig.add_subplot(2, 3, i + 1 + (len(titles)))
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
