import theano
import numpy
x = theano.tensor.matrix('x')
f = theano.function([x], (x ** 2).shape)
theano.printing.debugprint(f)
x = theano.tensor.matrix('x')
y = theano.tensor.matrix('y')
z = theano.tensor.join(0, x, y)
xv = numpy.random.rand(5, 4)
yv = numpy.random.rand(3, 3)

f1 = theano.function([x, y], z.shape)
theano.printing.debugprint(f)