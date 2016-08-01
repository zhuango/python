from sklearn import datasets, svm, cross_validation
import numpy as np

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

kfold = cross_validation.KFold(n = len(X_digits), k = 3)