import numpy
import pylab as pl
import numpy as np
import scipy
from scipy.optimize import minimize, rosen, rosen_der
import scipy.io as sio
def loadData(filename):
    data=sio.loadmat(filename)
    X=data['X']  
    y=data['y']
    return (X, y)

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

def costFunction(thetaFormat,X, Y, lamda):
    theta = thetaFormat.reshape((len(thetaFormat), 1))
    m = len(Y);
    n = len(theta)
    J = 0;

    z = sigmoid(numpy.dot(X, theta));

    J = 1.0 / m * numpy.sum(-Y * numpy.log(z) - (1 - Y) * numpy.log(1 - z)) + lamda / (2.0 * m) * numpy.sum(theta * theta)

    return J
def logicticRegressionJac(thetaFormat, X, Y, lamda):
    theta = thetaFormat.reshape((len(thetaFormat), 1))
    m = len(Y);
    n = len(theta);
    grad = numpy.zeros(n)

    z = sigmoid(numpy.dot(X, theta));

    for i in range(n - 1):
        grad[i] = 1.0 / m * numpy.sum((z - Y) * X[::, i:i+1]) + lamda / m * theta[i]
    grad[n - 1] = 1.0 / m * numpy.sum((z - Y) * X[::, n - 1:n])

    return grad
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

def formatY(Y, c):
    formY = numpy.zeros((len(Y), 1))
    for j in range(m):
        if Y[j, 0] == c:
            formY[j, 0] = 1
        else:
            formY[j, 0] = 0
    return formY
if __name__ == "__main__":

    (X, Y) = loadData('ex3data1.mat')

    #plotData(X, Y)
    m = len(X);
    bias = numpy.ones((m, 1));
    X = numpy.append(X, bias, 1);
    #X = mapFeature(X[::,0:1], X[::, 1:2], 6)#already add the bias column.
    m, n = numpy.shape(X)

    alpha = 0.01
    lamda = 0.1
    numLabels = 10;
    # for i in range(numberOfIter):
    #     cost, grad = costFunctionOld(theta, X, Y, lamda)
    #     theta = theta - alpha * grad;
    #     # print("cost: " + str(cost))
    #     # print("grad: " + str(grad))
    allTheta = numpy.zeros((numLabels, n));
    print(X[0].shape)
    print(allTheta.shape)
    for i in range(numLabels):
        theta = numpy.zeros((n, 1));
        result = minimize(costFunction,theta, (X, formatY(Y, i + 1), lamda),jac =logicticRegressionJac, method='CG', tol = 1e-10, options={'maxiter': 50, 'disp': True});
        allTheta[i] = result.x;
    
    correct = 0
    for i in range(m):
        if Y[i, 0] == numpy.argmax(sigmoid(numpy.dot(X[i], numpy.transpose(allTheta)))) + 1:
            correct += 1
    print(allTheta)
    print("Accuracy: " + str(float(correct) / float(m)))