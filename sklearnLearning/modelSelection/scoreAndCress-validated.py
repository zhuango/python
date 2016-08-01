from sklearn import datasets, svm

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
score = svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])
print(score)


# split the data in folds that we use for training and testing
# KFold cross validation
import numpy as np
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()
for k in range(3):
    X_train = list(X_folds)
    # pop out the element with index k.
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)

    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)

    socres.append(svc.fit(X_train, y_train).score(X_test, y_test))

print(scores)