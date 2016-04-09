import theano
import theano.tensor as T

x = T.dvector('x')
y = x ** 2
print(y.shape[0])
J, updates = theano.scan(lambda i, y, x: T.grad(y[i], x), sequences = T.arange(y.shape[0]), non_sequences = [y, x])
f = theano.function([x], J, updates= updates)
print(f([8, 4]))
