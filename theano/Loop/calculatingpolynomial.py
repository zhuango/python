import theano
import theano.tensor as T
import numpy

# xishu
coefficients = theano.tensor.vector("coefficients")
x = T.scalar("x")

max_coefficients_supported = 10000

components, updates = theano.scan(fn = lambda coefficient, power, free_variable: coefficient *(free_variable**power),
                                  sequences=[coefficients, theano.tensor.arange(max_coefficients_supported)],
                                  outputs_info=None,
                                  non_sequences = x
                                  )
                                  theano.tensor.arange(max_coefficients_supported)
polynomial = components.sum()

calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=polynomial)

test_coefficients = numpy.asarray([1, 0, 2], dtype = numpy.float32)
test_value = 3
print(calculate_polynomial(test_coefficients, test_value))
print(1.0 * (3 ** 0) + 0.0 * (3 ** 1) + 2.0 * (3 ** 2))
