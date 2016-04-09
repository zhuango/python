from theano import tensor as T
from theano.ifelse import ifelse
import theano, time, numpy

a, b = T.scalars('a', 'b')
x, y = T.matrices('x', 'y')

z_switch = T.switch(T.lt(a, b), T.mean(x), T.mean(y))
z_lazy = ifelse(T.lt(a, b), T.mean(x), T.mean(y))

f_switch = theano.function([a, b, x, y], z_switch,
mode = theano.Mode(linker = 'vm'), allow_input_downcast=True)

f_lazyifelse = theano.function([a, b, x, y], z_lazy,
mode = theano.Mode(linker = 'vm'), allow_input_downcast=True)

val1 = 0.
val2 = 1.

big_mat1 = numpy.ones((10000, 1000))
big_mat2 = numpy.ones((10000, 1000))

n_times = 10

tic = time.clock()
for i in range(n_times):
    f_switch(val1, val2, big_mat1, big_mat2)
print('time spent evaluating one value %f sec' % (time.clock() - tic))

tic = time.clock()
for i in range(n_times):
    f_lazyifelse(val1, val2, big_mat1, big_mat2)
print('time spent evaluating one value %f sec' % (time.clock() - tic))
