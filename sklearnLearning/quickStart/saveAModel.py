from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
models = clf.fit(X, y)

import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)

result = clf2.predict(X[0:1])
print(result)

from sklearn.externals import joblib
files = joblib.dump(clf, 'filename.pkl')
print(files)
clf = joblib.load('filename.pkl')
