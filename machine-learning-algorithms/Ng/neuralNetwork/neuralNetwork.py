import numpy
import scipy
import scipy.io as sio

def sigmoid(z):
    result = numpy.zeros(z.shape);
    result = 1.0 / (1.0 + numpy.exp(-z));
    return result;
def sigmoidGradient(z):
    g = numpy.zeros(size(z));
    g = sigmoid(z) * (1 - sigmoid(z));
    return g
    
def predict(theta1, theta2, X):
    m = len(X);
    num_labels = len(theta2);

    p =numpy.ones((numpy.size(X, 0), 1));

    bias = numpy.ones((m,));
    tmpX = numpy.insert(X, 0, bias, 1)

    hidden = sigmoid(numpy.dot(theta1, numpy.transpose(tmpX)));
    hidden = numpy.insert(numpy.transpose(hidden), 0, bias, 1)
    indexOfMax = numpy.argmax(sigmoid(numpy.dot(theta2, numpy.transpose(hidden))), 0) + 1;
    
    return indexOfMax;

if __name__ == "__main__":
    input_layer_size= 400;
    hidden_layer_size = 25;
    num_labels = 10;

    data = sio.loadmat('ex3data1.mat');
    X = data['X'];
    Y = data['y'];
    m = len(X);

    weights = sio.loadmat('ex3weights.mat');
    theta1 = weights['Theta1'];
    theta2 = weights['Theta2'];

    pred = predict(theta1, theta2, X);
    correct = 0
    for i in range(m):
        if Y[i, 0] == pred[i]:
            correct += 1
    print("Accuracy: " + str(float(correct) / float(m)))