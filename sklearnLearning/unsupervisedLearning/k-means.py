#!/usr/bin/python3
from sklearn import cluster, datasets
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

k_means = cluster.KMeans(n_clusters=3)
models = k_means.fit(X_iris, y_iris)
print(models)

print(k_means.labels_[::10])
print(y_iris[::10])