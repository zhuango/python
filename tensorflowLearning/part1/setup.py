from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batched = total_series_length // batch_size // truncated_backprop_length

def generateData():
    x = np.array(np.random.choice(2, total_series_length, p = [0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return (x, y)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32,   [batch_size, truncated_backprop_length])

W  = tf.Variable(np.random.rand([state_size + 1, state_size], dtype=tf.float32))
b  = tf.Variable(np.zeros((1, state_size)), dtype=tf.float23)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype = tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)
