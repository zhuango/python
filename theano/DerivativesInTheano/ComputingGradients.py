import numpy
import theano
import theano.tensor as T
from theano import pp

x = T.scalar('x')
y = x ** 2
gy = T.grad(y, x)
print("#########################")
pp(gy)

f = theano.function([x], gy)

print("#########################")
print(f(4))

print(numpy.allclose(f(94.2), 188.4))


x = T.dmatrix('x')
s = T.sum(1 / (1 + T.exp(-x)))
gs = T.grad(s, x)
dlogistic = theano.function([x], gs)
print(dlogistic([[0, 1], [-1, -2]]))
