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
    pca = PCA(n_components=n_compo)
    x_reduced = pca.fit_transform(X)

    # compute the eigenvalues and eigenvectors and reduced data to the number of components that have eigenvalue
    # bigger than 1 (more representative, has bigger variance)
    biggest_eigenvalues = [i for i in pca.singular_values_ if i > 1]
    x_pca = x_reduced[:, :len(biggest_eigenvalues)]
    eigenvectors = pca.components_[:len(biggest_eigenvalues), :]

    # plot the components and eigenvalues to understand the number of optimal components
    x = np.arange(len(pca.singular_values_))
    labels = [str(i + 1) + 'º Component' for i in list(x)]
    plt.bar(x, pca.singular_values_)
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

    ipca = IncrementalPCA(n_components=n_compo)
    x_reduced_ipca = ipca.fit_transform(X)

    # compute the eigenvalues and eigenvectors and reduced data to the number of components that have eigenvalue
    # bigger than 1 (more representative, has bigger variance)
    biggest_eigenvalues_ipca = [i for i in ipca.singular_values_ if i > 1]
    x_ipca = x_reduced_ipca[:, :len(biggest_eigenvalues_ipca)]
    eigenvectors_ipca = ipca.components_[:len(biggest_eigenvalues_ipca), :]

    # plot the components and eigenvalues to understand the number of optimal components
    xipca = np.arange(len(ipca.singular_values_))
    labels_ipca = [str(i + 1) + 'º Component' for i in list(xipca)]
    plt.title('Incremental PCA - ' + dataset_name + ' data set')
    plt.ylabel('Eigenvalue')
    plt.bar(xipca, ipca.singular_values_)
    plt.xticks(xipca, labels_ipca)
    plt.show()

    return dict(eigenvalues=biggest_eigenvalues_ipca, eigenvectors=eigenvectors_ipca, db=x_ipca)


def compare_sklearn_results(our_pca_data, pca_data, ipca_data):
    print("Our PCA:\n", our_pca_data)
    print("PCA: \n", pca_data['db'])
    print("PCA incremental\n", ipca_data['db'])
