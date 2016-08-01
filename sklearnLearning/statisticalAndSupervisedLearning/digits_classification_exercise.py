from sklearn import datasets, neighbors, linear_model
import numpy

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

trainSize = int(len(X_digits) * 0.9)

X_train = X_digits[:trainSize]
y_train = y_digits[:trainSize]

X_test  = X_digits[trainSize:]
y_test  = y_digits[trainSize:]

labels = numpy.unique(y_digits)

lrs = []
for label in labels:
    lr = linear_model.LogisticRegression()
    lr.fit(X_train, numpy.array(y_train==label, dtype=int))
    lrs.append(lr)

results = []
for lr in lrs:
    results.append(lr.predict(X_test))
resultMatrix = numpy.array(results)
print(resultMatrix.shape)
finalResult = numpy.argmax(resultMatrix, axis=0)
print("logistic regression: ")
print(numpy.mean(numpy.array(finalResult==y_test, dtype=int)))




kn = neighbors.KNeighborsClassifier(n_neighbors=len(labels))
kn.fit(X_train, y_train)
result = kn.predict(X_test)
print("k neighbors: ")
print(numpy.mean(numpy.array(result==y_test, dtype=int)))
print(y_test)


knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()

print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
print('LogisticRegression score: %f'
      % logistic.fit(X_train, y_train).score(X_test, y_test))
