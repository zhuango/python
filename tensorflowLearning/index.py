import tensorflow as tf
import numpy

x = tf.placeholder(tf.float32, [4, 4], name='x')
test = 10
y = x[0, 1] + 10 + test
maxIndex = tf.argmax(x, 1)
sess = tf.Session()
inputX = numpy.arange(16).reshape((4, 4))
y, maxIndex = sess.run([y, maxIndex], feed_dict={x:inputX})
print(y)
print(maxIndex)