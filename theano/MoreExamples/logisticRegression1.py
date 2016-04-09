import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
import time 

#rng = numpy.random
rng = RandomStreams(seed=234)

N = 400     ## training sample size
feats = 784 #28 * 28, number of input variables

D = (rng.normal((N, feats)), rng.uniform(size = (1,N), low = 0, high = 2))
training_steps = 10000

#Declare Theano symbolic varibles
x = T.matrix("x")
y = T.vector("y")


# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = (rng.normal((1, feats)))

#initialize the bias term
b = theano.shared(0., name = "b")
print("Initial model:")
print(w.)
print(b.get_value())

t0 = time.time()

#Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b)) # 1 \ (1 + e^-(x*w - b))
prediction = p_1 > 0.5 #The prediction thresholded
xent = -y * T.log(p_1) - (1 - y) * T.log(1-p_1)
cost = xent.mean() + 0.01 * (w ** 2).sum()
gw,gb = T.grad(cost, [w, b])

#Compile
train = theano.function(
    inputs=[x, y], 
    outputs = [prediction, xent],
    updates = ((w, w - 0.1 * gw), (b, b - 0.1 * gb)),
    allow_input_downcast=True)
predict = theano.function(inputs = [x], outputs = prediction, allow_input_downcast=True)

#Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])


print("Final model:")
print(w.get_value())
print(b.get_value())
print("Target values for D:")
print(D[1])
print("prediction in D:")
print(predict(D[0]))

t1 = time.time()
print("Looping took %f seconds" % (t1 - t0))
