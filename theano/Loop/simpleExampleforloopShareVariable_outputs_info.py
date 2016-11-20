import theano
import theano.tensor as T
import numpy

def power_of_2(previous_power, A):
    return previous_power * 2, A * 2
max_value = T.scalar("max_")
A = T.tensor3("A")
values, _ = theano.scan(power_of_2,
                        outputs_info = [T.constant(1.), A],
                        n_steps = 8)
f = theano.function([max_value, A], values, on_unused_input='ignore')
print(f(1,numpy.asarray([[[1], [2], [3]],[[4], [5], [6]]] , dtype=theano.config.floatX))[1])