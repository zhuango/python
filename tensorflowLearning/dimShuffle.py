import numpy
import theano
import theano.tensor as T

b_values = numpy.ones((10,), dtype=theano.config.floatX)
b = theano.shared(value=b_values, borrow=True)
shuffleB = b.dimshuffle('x', 0, 'x', 'x')

f = theano.function([], shuffleB)
print(numpy.shape(f()))

import tensorflow as tf
b_values = numpy.ones((10,), dtype=theano.config.floatX)
b = tf.Variable(initial_value=b_values, name='b', dtype=tf.float32)
shuffleB = tf.expand_dims(tf.expand_dims(tf.expand_dims(b, dim=0),2), 3)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(numpy.shape(sess.run(shuffleB)))