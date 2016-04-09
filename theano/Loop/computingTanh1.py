import theano
import theano.tensor as T
import numpy as np

#define tensor variables
X = T.vector("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")
U = T.matrix("U")
Y = T.matrix("Y")
V = T.matrix("V")
P = T.matrix("P")

results, updates = theano.scan(lambda y, p, x_tm1: T.tanh(T.dot(x_tm1, W) + T.dot(y, U) + T.dot(p, V)),
    sequences = [Y, P[::-1]], outputs_info = [X])
    compute_seq = theano.function(inputs = [X, W, Y, U, P, V], outputs = results)
    
#test values
x = np.zeros((2), dtype = theano.config.floatX)
x[1] = 1
w = np.ones((2, 2), dtype = theano.config.floatX)
y = np.ones((5, 2), dtype = theano.config.floatX)
y[0, :] = -3
u = np.ones((2, 2), dtype = theano.config.floatX)
p = np.ones((5, 2), dtype = theano.config.floatX)
p[0,:] = 3
v = np.ones((2, 2), dtype = theano.config.floatX)

print(compute_seq(x, w, y, u, p, v))
#comparison with numpy
x_res = np.zeros((5, 2), dtype = theano.config.floatX)
x_res[0] = np.tanh(x.dot(w), y[0].dot(u) + p[4].dot(v))
for i in range(1, 5):
    x_res[i] = np.tanh(x_res[i - 1].dot(w) + y[i].dot(u) + p[4-i].dot(v))
print(x_res)