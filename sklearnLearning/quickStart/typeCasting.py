#! /usr/bin/python3
import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype = 'float32')
print(X.dtype)

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.dtype)

from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
clf = SVC()
models = clf.fit(iris.data, iris.target)
print(models)
result = list(clf.predict(iris.data[:3]))
print(result)

models = clf.fit(iris.data, iris.target_names[iris.target])
print(models)
result = list(clf.predict(iris.data[:3]))
print(result)