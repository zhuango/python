import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

config = tf.ConfigProto()
sess = tf.Session(config=config)

X = np.random.randn(2,10,8)

# The second example is of length 6 

#X[1,6:] = 0

X_lengths = [10,6]

#cell = tf.nn.rnn_cell.LSTMCell(num_units=20)
cell = tf.nn.rnn_cell.GRUCell(num_units=20)

outputs,states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw=cell,dtype=tf.float64,sequence_length=X_lengths,inputs=X)

output_fw, output_bw = outputs
states_fw, states_bw = states

sess.run(tf.global_variables_initializer())
print(sess.run([output_fw, states_fw]))
