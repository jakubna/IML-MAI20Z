import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA


def pca_sklearn(X, dataset_name, n_compo):
    """ function to compute PCA using sklearn algorithm
        X: 2D data array of size (rows, features).
        dataset_name: string (name of data set to set plot title).
        n_compo: number of components that we want to reduce the X dataset.
        return: eigenvectors, eigenvalues , reduced data set
    """

    pca = PCA(n_components=X.shape[1])
    x_reduced = pca.fit_transform(X)
    print('SKLearn PCA')
    print('All Eigenvalues:\n ', pca.explained_variance_)
    print('All Eigenvectors:\n ', pca.components_)
    eig_map = list(zip(pca.explained_variance_, pca.components_))
    print('eig_map:\n',eig_map)

    # compute the eigenvalues and eigenvectors and reduced data to the number of components that:
    # 1) was required
    # 2) represents n% of the explained variance
    # 3) is equal to number of features
    if type(n_compo) == int:
        k=n_compo
    elif type(n_compo) ==float:
        k = np.searchsorted(pca.explained_variance_ratio_.cumsum(),n_compo) + 1
    else:
        k=X.shape[1]
    print('k = ',k)
    biggest_eigenvalues = pca.explained_variance_[:k]
    x_pca = x_reduced[:, :k]
    eigenvectors = pca.components_[:k, :]

    print('K first eigenvalues:\n', biggest_eigenvalues)
    print('K eigenvectors:\n', eigenvectors)


    # plot the components and eigenvalues to understand the number of optimal components
    x = np.arange(len(pca.explained_variance_))
    labels = [str(i + 1) + 'º Component' for i in list(x)]
    plt.bar(x, pca.explained_variance_)
    plt.title('PCA - ' + dataset_name + ' data set')
    plt.ylabel('Eigenvalue')
    plt.xticks(x, labels)
    plt.show()

    return dict(eigenvalues=biggest_eigenvalues, eigenvectors=eigenvectors, db=x_pca)


def ipca_sklearn(X, dataset_name, n_compo):
    """ function to compute PCA using sklearn algorithm
        X: 2D data array of size (rows, features).
        dataset_name: string (name of data set to set plot title).
        n_compo: number of components that we want to reduce the X dataset.
        return: eigenvectors, eigenvalues , reduced data set
    """

    ipca = IncrementalPCA(X.shape[1])
    x_reduced_ipca = ipca.fit_transform(X)
    print('SKLearn Incremental PCA')
    print('All Eigenvalues:\n ', ipca.explained_variance_)
    print('All Eigenvectors:\n ', ipca.components_)
    eig_map = list(zip(ipca.explained_variance_, ipca.components_))
    print('eig_map:\n',eig_map)
    # compute the eigenvalues and eigenvectors and reduced data to the number of components that:
    # 1) was required
    # 2) represents n% of the explained variance
    # 3) is equal to number of features
    # bigger than 1 (more representative, has bigger variance)

    if type(n_compo) == int:
        k=n_compo
    elif type(n_compo) ==float:
        k = np.searchsorted(ipca.explained_variance_ratio_.cumsum(),n_compo) + 1
    else:
        k=X.shape[1]
    print('k = ', k)
    biggest_eigenvalues_ipca = ipca.explained_variance_[:k]
    x_ipca = x_reduced_ipca[:, :k]
    eigenvectors_ipca = ipca.components_[:k, :]

    print('K first eigenvalues:\n', biggest_eigenvalues_ipca)
    print('K eigenvectors:\n', eigenvectors_ipca)

    # plot the components and eigenvalues to understand the number of optimal components
    xipca = np.arange(len(ipca.explained_variance_))
    labels_ipca = [str(i + 1) + 'º Component' for i in list(xipca)]
    plt.title('Incremental PCA - ' + dataset_name + ' data set')
    plt.ylabel('Eigenvalue')
    plt.bar(xipca, ipca.explained_variance_)
    plt.xticks(xipca, labels_ipca)
    plt.show()

    return dict(eigenvalues=biggest_eigenvalues_ipca, eigenvectors=eigenvectors_ipca, db=x_ipca)
