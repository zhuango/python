import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

X = np.array([[-1, -1, -1], [-2, -1, -2], [-3, -2, -3], [1, 1, 1], [2, 1, 2], [3, 2, 3]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.components_)

print("++++++++++")
x_mean = np.mean(X, 0)
X = X - x_mean
X = np.dot(X.T, X)
U, s, V = np.linalg.svd(X)
print(U[:, 0:2].T)
print(s)