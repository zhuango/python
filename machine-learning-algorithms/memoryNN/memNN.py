import numpy as np
import tensorflow as tf

batchSize = 100
vectorLength = 300
sentMaxLength = 20
hopNumber = 3
classNumber = 2

test_contxtWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, sentMaxLength])
test_aspectWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, 1])
test_labels_placeholder      = tf.placeholder(tf.float32, [batchSize, 1])

test_position_placeholder    = tf.placeholder(tf.float32, [batchSize, 1, sentMaxLength])
test_sentLength_placeholder  = tf.placeholder(tf.float32, [batchSize, 1])

contxtWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, sentMaxLength])
aspectWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, 1])
labels_placeholder      = tf.placeholder(tf.float32, [batchSize, 1])

position_placeholder    = tf.placeholder(tf.float32, [batchSize, 1, sentMaxLength])
sentLength_placeholder  = tf.placeholder(tf.float32, [batchSize, 1])

attention_W = tf.Variable(np.random.rand(1, 2 * vectorLength), dtype = tf.float32)
attention_b = tf.Variable(np.tandom.rand(1), dtype = tf.float32)

linearLayer_W = tf.Variable(np.random.rand(vectorLength, vectorLength), dtype=tf.float32)
linearLayer_b = tf.Variable(np.random.rand(vectorLength, 1), dtype = tf.float32)

softLayer_W = tf.Variable(np.random.rand(classNumber, vectorLength), dtyp= tf.float32)
softLayer_b = tf.Variable(np.random.rand(classNumber, 1))

Vaspect = aspectWords_placeholder

for i in range(hopNumber):
    Vi = 1 - position_placeholder / sentLength_placeholder - (hopNumber / vectorLength) * (1 - 2 * position_placeholder / sentLength_placeholder)
    Mi = Vi * contxtWords_placeholder
    gi = tf.tanh(tf.matmul(attention_W, tf.concat(Mi, Vaspect)) + attention_b)
    alpha = tf.nn.softmax(gi)
    Vaspect = tf.summary(alpha * Mi) + (tf.matmul(linearLayer_W, Vaspect) + linearLayer_b)

losses = tf.nn.softmax_cross_entropy_with_logits()