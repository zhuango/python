import tensorflow as tf
import numpy as np
from HiddenLayer import HiddenLayer

randomGenerator = np.random.RandomState(23333)

input = tf.placeholder(dtype=tf.float32, shape=[4, 10], name='input')
layer = HiddenLayer(randomGenerator, input, 4, 2)
output = layer.output

sess = tf.Session()
realOutput = sess.run(output, feed_dict={input:np.arange(40).reshape(4, 10)})
print(realOutput)

