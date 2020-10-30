import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot2d(x: np.ndarray, y, titles):
    """ the function aims to reduce the features to two dimensions using PCA method and plot the clusters
        x: 2D data array of size (rows, features).
        y_predict: results labels.
        title: graphic title (example: adults dataset using KMeans).
    """
    fig = plt.figure(figsize=(20, 20))
    for i in range(0, len(titles)):
        y_predict = y[i]
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(x)
        final_df = pd.DataFrame(data=principal_components,
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
            ax.scatter(final_df.loc[indices2keep, 'principal component 1'],
                       final_df.loc[indices2keep, 'principal component 2'],
                       c=color,
                       s=50)
        ax.legend(targets)
        ax.grid()
    plt.show()


