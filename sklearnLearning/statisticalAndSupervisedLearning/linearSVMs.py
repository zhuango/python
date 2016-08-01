from sklearn import svm
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

iris_X = iris.data[iris_y != 0, :2]
iris_y = iris.target[iris_y != 0]

print(np.unique(iris_y))
np.random.seed(0)
trainSize = int(len(iris_X) * 0.9)
iris_X_train = iris_X[:trainSize]
iris_y_train = iris_y[:trainSize]
iris_X_test = iris_X[trainSize:]
iris_y_test = iris_y[trainSize:]



svc = svm.SVC(kernel='linear')
models = svc.fit(iris_X_train, iris_y_train)
print(models)
print(svc.predict(iris_X_test))
print(iris_y_test)