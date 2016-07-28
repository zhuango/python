from sklearn import datasets
import numpy as np

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.data[:-20]
diabetes_y_test  = diabetes.data[-20:]

# linear models: y = X * beta + b

from sklearn import linear_model
regr = linear_model.LinearRegression()
models = regr.fit(diabetes_X_train, diabetes_y_train)
print(models)
print(regr.coef_)

mean =np.mean((regr.predict(diabetes_X_test) - diabetes_y_test)**2)
print(mean)

score = regr.score(diabetes_X_test, diabetes_y_test)
print(score)