from sklearn import cross_validation, datasets, linear_model
from sklearn.grid_search import GridSearchCV
import numpy as np

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]

lasso = linear_model.Lasso()
alphas = np.logspace(-4, -.5, 30)

clf = GridSearchCV(estimator = lasso, param_grid=dict(alpha = alphas), n_jobs = -1)
clf.fit(X, y)
print(clf.best_score_)
print(clf.best_estimator_.alpha)
clf.score(X, y)