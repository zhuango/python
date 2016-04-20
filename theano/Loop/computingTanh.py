import theano
import theano.tensor as T
import numpy as np
import time
# Computing tanh(x(t).dot(W) + b) elementwise

# define the tensor variables
X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")

results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym), sequences = X)

compute_elementwise = theano.function(inputs=[X, W, b_sym], outputs=results)

# test values
# eye Return a 2-D array with ones on the diagonal and zeros elsewhere.
x = np.eye(2, dtype=theano.config.floatX)
# ones Return a new array of given shape and type, filled with ones.
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones(2, dtype=theano.config.floatX)
b[1] = 2


t0 = time.time()
print(compute_elementwise(x, w, b))
t1 = time.time()
print("compute_elementwise took %f sec." % (t1 - t0))

t0 = time.time()
# comparison with numpy
print(np.tanh(x.dot(w) + b))
t1 = time.time()
print("np.tanh took %f sec." % (t1 - t0))