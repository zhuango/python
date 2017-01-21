import numpy as np
import tensorflow as tf
from HiddenLayer import HiddenLayer

randomGenerator = np.random.RandomState(23333)

input = tf.placeholder(dtype=tf.float32, shape=[10, 4], name='input')
layer = HiddenLayer(randomGenerator, input, 4, 2)
output = layer.output

sess = tf.Session()
sess.run(tf.initialize_all_variables()) # initialize variables

realOutput = sess.run(output, feed_dict={input:np.asarray(np.arange(40).reshape(10, 4), dtype=np.float)})
print(realOutput)

# [[ 0.70869356 -0.36734974]
#  [ 0.97741109  0.98196203]
#  [ 0.99847019  0.99992317]
#  [ 0.99989742  0.99999964]
#  [ 0.99999321  1.        ]
#  [ 0.99999952  1.        ]
#  [ 0.99999994  1.        ]
#  [ 1.          1.        ]
#  [ 1.          1.        ]
#  [ 1.          1.        ]]