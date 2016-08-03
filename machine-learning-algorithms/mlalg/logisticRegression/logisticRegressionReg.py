import numpy
import pylab as pl
def loadData(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(list(map(float, line.split(','))))
    return numpy.array(data)

def plotData(X, Y):
    x = X[::, 0:1]
    y = X[::, 1:2]
    for i in range(0, len(y)):
        if Y[i, 0] == 0:
            pl.plot(x[i, 0], y[i, 0], 'ok')# use pylab to plot x and yield
        else:
            pl.plot(x[i, 0], y[i, 0], 'xk') 
    pl.show()# show the plot on the screen

def sigmoid(z):
    result = numpy.zeros(z.shape);
    result = 1.0 / (1.0 + numpy.exp(-z));
    return result;

def costFunction(theta,X, Y, lamda):
    m = len(Y);
    n = len(theta)
    J = 0;
    grad = numpy.zeros((n, 1))

    z = sigmoid(numpy.dot(X, theta));
    J = 1.0 / m * numpy.sum(-Y * numpy.log(z) - (1 - Y) * numpy.log(1 - z)) + lamda / (2.0 * m) * numpy.sum(theta * theta)

    for i in range(n - 1):
        grad[i, 0] = 1.0 / m * numpy.sum((z - Y) * X[::, i:i+1]) + lamda / m * theta[i, 0]
    grad[n - 1, 0] = 1.0 / m * numpy.sum((z - Y) * X[::, n - 1:n])
    return (J, grad)

def mapFeature(X1, X2, degree):
    #degree = 6;
    # Arithmetic sequence sum would be the number of feature.
    # and one bias column.
    featureNumber = 0;
    out = numpy.ones((len(X1), (2 + degree + 1) * degree / 2 + 1))
    for i in range(1, degree + 1):
        for j in range(0, i+1):
            out[::, featureNumber:featureNumber + 1] = numpy.power(X1, i - j) * numpy.power(X2, j);
            featureNumber += 1
    return out;
if __name__ == "__main__":

    data = loadData('ex2data2.txt')
    # for i in range(len(data)):
    #     data[i].insert(0, 0);
    X = numpy.array(data[::, 0:2]) # first 2 column represents X.
    Y = numpy.array(data[::, 2:3]) # last column represents Y.

    plotData(X, Y)

    # bias = numpy.ones((m, 1));
    # X = numpy.append(X, bias, 1);
    X = mapFeature(X[::,0:1], X[::, 1:2], 6)#already add the bias column.
    m, n = numpy.shape(X)

    theta = numpy.zeros((n, 1))
    alpha = 0.01
    lamda = 1
    numberOfIter = 60000
    for i in range(numberOfIter):
        cost, grad = costFunction(theta, X, Y, lamda)
        theta = theta - alpha * grad;
        # print("cost: " + str(cost))
        # print("grad: " + str(grad))
    
    correct = 0
    for i in range(m):
        if Y[i, 0] == 1 and numpy.dot( X[i], theta) >= 0:
            correct += 1
        elif Y[i, 0] == 0 and numpy.dot( X[i], theta) < 0:
            correct += 1 
    print(theta)
    print("Accuracy: " + str(float(correct) / float(m)))