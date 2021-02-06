import numpy as np
import matplotlib.pyplot as plt


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.std = None
        self.cov = None
        self.eigen_values = None
        self.components = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X = X - self.mean
        X = X / self.std
        self.cov = np.cov(X.T)
        self.eigen_values, eigen_vectors = np.linalg.eig(self.cov)
        eigen_vectors = eigen_vectors.T
        indices = np.argsort(self.eigen_values)[::-1]
        self.eigen_values = self.eigen_values[indices]
        eigen_vectors = eigen_vectors[indices]
        self.components = eigen_vectors[:self.n_components]

    def transform(self, X):
        X = X - self.mean
        X = X / self.std
        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def plot_cov_matrix(self):
        cov_img = plt.matshow(self.cov, cmap=plt.cm.Reds)
        plt.colorbar(cov_img, ticks=[-1, 0, 1])
        for x in range(self.cov.shape[0]):
            for y in range(self.cov.shape[1]):
                plt.text(x, y, "%0.2f" % self.cov[x, y], size=10,
                color="black", ha="center", va="center")
        plt.show()

    def explained_variance(self):
        return self.eigen_values

    def explained_variance_ratio(self):
        return self.eigen_values / np.sum(self.eigen_values)

    def plot_cumulative_explained_variance_ratio(self):
        plt.plot(
            list(range(1, len(np.cumsum(self.explained_variance_ratio())) + 1)),
            np.cumsum(self.explained_variance_ratio())
        )
        plt.xlabel('Nombre de Composantes')
        plt.ylabel('% Explained Variance')
        plt.title('PCA Somme Cumulative Variance')
        plt.show()
