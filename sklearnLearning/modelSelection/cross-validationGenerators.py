from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

svc = svm.SVC(C=1, kernel='linear')
k_fold = cross_validation.KFold(n = len(X_digits), n_folds = 3)
for train_indices, test_indices in k_fold:
    print("Train: %s | test: %s" % (train_indices, test_indices))

results = [svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]) for train , test in k_fold]
print(results)


# n_jobs=-1 means that the computation will be dispatched on all the CPUs of the computer.
results = cross_validation.cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs = -1)

print(results)