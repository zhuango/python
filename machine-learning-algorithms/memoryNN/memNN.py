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
test_sentLength_placeholder  = tf.placeholder(tf.float32, [batchSize, 1, 1])

contxtWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, sentMaxLength])
aspectWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, 1])
labels_placeholder      = tf.placeholder(tf.float32, [batchSize, 2])
position_placeholder    = tf.placeholder(tf.float32, [batchSize, 1, sentMaxLength])
sentLength_placeholder  = tf.placeholder(tf.float32, [batchSize, 1, 1])

attention_W = tf.Variable(np.random.rand(1, 2 * vectorLength), dtype = tf.float32)
attention_b = tf.Variable(np.random.rand(1), dtype = tf.float32)

linearLayer_W = tf.Variable(np.random.rand(vectorLength, vectorLength), dtype=tf.float32)
linearLayer_b = tf.Variable(np.random.rand(vectorLength, 1), dtype = tf.float32)

softmaxLayer_W = tf.Variable(np.random.rand(classNumber, vectorLength), dtype= tf.float32)
softmaxLayer_b = tf.Variable(np.random.rand(classNumber, 1), dtype= tf.float32)

vaspect = aspectWords_placeholder

for i in range(hopNumber):
    Vi = 1 - position_placeholder / sentLength_placeholder - (hopNumber // vectorLength) * (1 - 2 * position_placeholder / sentLength_placeholder)
    Mi = Vi * contxtWords_placeholder
    expanded_vaspect = vaspect
    for j in range(sentMaxLength - 1):
        expanded_vaspect = tf.concat(2, [expanded_vaspect, vaspect])
    attentionInputs = tf.unpack(tf.concat(1, [Mi, expanded_vaspect]), axis=0)
    gi = tf.pack([tf.tanh(tf.matmul(attention_W, input) + attention_b) for input in attentionInputs])
    alpha = tf.nn.softmax(gi)    
    linearLayerOut = tf.pack([tf.matmul(linearLayer_W, input) + linearLayer_b for input in tf.unpack(vaspect, axis=0)])
    vaspect = tf.reduce_sum(alpha * Mi, 2, True) + linearLayerOut

linearLayerOut = tf.pack([tf.matmul(softmaxLayer_W, input) + softmaxLayer_b for input in tf.unpack(vaspect, axis=0)])
calssification = tf.reduce_sum(tf.nn.softmax(linearLayerOut), 2)

losses = tf.nn.softmax_cross_entropy_with_logits(calssification, labels_placeholder)
lossesToMin = tf.reduce_sum(losses)

