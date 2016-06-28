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

def costFunction(theta,X, Y):
    m = len(Y);
    J = 0;
    grad = numpy.zeros((len(theta), 1))

    z = sigmoid(numpy.dot(X, theta));
    J = 1.0 / m * numpy.sum(-Y * numpy.log(z) - (1 - Y) * numpy.log(1 - z))

    for i in range(len(theta)):
        grad[i] = 1.0 / m * numpy.sum((z - Y) * X[::, i:i+1])
    return (J, grad)
if __name__ == "__main__":

    data = loadData('ex2data1.txt')
    # for i in range(len(data)):
    #     data[i].insert(0, 0);
    X = numpy.array(data[::, 0:2]) # first 2 column represents X.
    Y = numpy.array(data[::, 2:3]) # last column represents Y.

    plotData(X, Y)

    m, n = numpy.shape(X)
    bias = numpy.ones((m, 1));
    X = numpy.append(X, bias, 1);

    theta = numpy.zeros((n + 1, 1))
    alpha = 0.001
    numberOfIter = 60000
    for i in range(numberOfIter):
        cost, grad = costFunction(theta, X, Y)
        theta = theta - alpha * grad;
        #print("cost: " + str(cost))
        #print("grad: " + str(grad))
    
    correct = 0
    for i in range(m):
        if Y[i, 0] == 1 and numpy.dot( X[i], theta) >= 0:
            correct += 1
        elif Y[i, 0] == 0 and numpy.dot( X[i], theta) < 0:
            correct += 1 
    print(theta)
    print("Accuracy: " + str(float(correct) / float(m)))