import matplotlib.pyplot as plt
from sklearn import datasets

from pca import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearn_pca_model

iris_dataset = datasets.load_iris()
orginal_data = iris_dataset.data
target = iris_dataset.target

#print(orginal_data[:20])
#print(target[:20])
#print(orginal_data.shape)
#print(target.shape)

print("PCA FROM SCRATCH:")
pca_from_scratch = PCA(n_components=2)
pca_from_scratch.fit(orginal_data)
transformed_data = pca_from_scratch.transform(orginal_data)
#transformed_data = pca_from_scratch.fit_transform(orginal_data)

pca_from_scratch.plot_cov_matrix()
pca_from_scratch.plot_cumulative_explained_variance_ratio()

print(pca_from_scratch.components)
print(pca_from_scratch.explained_variance())
print(pca_from_scratch.explained_variance_ratio())

print()
print("PCA SCIKIT-LEARN:")
sklearn_pca = sklearn_pca_model(n_components=2)
scaler = StandardScaler()
scaler.fit(orginal_data)
normalized_data = scaler.transform(orginal_data)
transformed_data_sklearn = sklearn_pca.fit_transform(normalized_data)

print(sklearn_pca.components_)
print(sklearn_pca.explained_variance_)
print(sklearn_pca.explained_variance_ratio_)

fig, axes = plt.subplots(1, 3, figsize=(15, 7))
axes[0].scatter(orginal_data[:, 0], orginal_data[:,1], c=target)
axes[0].set_xlabel('VAR1')
axes[0].set_ylabel('VAR2')
axes[0].set_title('Avant PCA')
axes[1].scatter(transformed_data[:, 0], transformed_data[:,1], c=target)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('PCA From Scratch')
axes[2].scatter(transformed_data_sklearn[:, 0], transformed_data_sklearn[:,1], c=target)
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')
axes[2].set_title('Scikit-Learn PCA')

plt.show()