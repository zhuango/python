import numpy as np
import tensorflow as tf

class HiddenLayer(object):
    def __init__(self, rng, input, inputSize, outputSize, W=None, b=None, activation=tf.tanh):
        self.input = input
        if W is None:
            WInitialValues = np.asarray(
                rng.uniform(
                    low  = -np.sqrt(6. / (inputSize + outputSize)),
                    high =  np.sqrt(6. / (inputSize + outputSize)),
                    size = (outputSize, inputSize)
                ),
                dtype = np.float
            )
            if activation == tf.sigmoid:
                WInitialValues *= 4
            W = tf.Variable(initial_value=WInitialValues, name='W', dtype=tf.float32)
            
        if b is None:
            bValues = np.zeros((outputSize,1), dtype=np.float)
            b = tf.Variable(initial_value=bValues, name='b', dtype=tf.float32)
        self.W = W
        self.b = b
        print(tf.matmul(self.W, self.input))
        print(self.b)
        tempOutput  = tf.matmul(self.W, self.input) + self.b
        self.output = (
            tempOutput if activation is None
            else activation(tempOutput)
        )
        self.params = [self.W, self.b]
