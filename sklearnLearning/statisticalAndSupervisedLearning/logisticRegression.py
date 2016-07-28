from sklearn import linear_model
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print(np.unique(iris_y))

# Split iris data in train and test datasets
# A random permutation, to split the data randomly

np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]


logistic = linear_model.LogisticRegression(C=1e5)

models = logistic.fit(iris_X_train, iris_y_train)

print(models)

result = logistic.predict(iris_X_test)
print(result)
print(iris_y_test)