import numpy as np
import tensorflow as tf

batchSize = 100
vectorLength = 300
sentMaxLength = 20
hopNumber = 1
classNumber = 2
num_epoches = 10

def generateData():
    contxtWords = np.array(np.random.rand(batchSize, vectorLength, sentMaxLength))
    aspectWords = np.array(np.random.rand(batchSize, vectorLength, 1))
    labels = np.array(np.random.choice(2, (batchSize, 2), p=[0.5, 0.5]))
    #labels      = np.zeros((batchSize, 2))
    # for i in range(len(labels)):
    #     if np.random.rand(1) > 0.5:
    #         labels[i][0] = 1
    #     else:
    #         labels[i][1] = 1
    sentLength  = np.zeros((batchSize, 1, 1)) + sentMaxLength

    position    = np.zeros((batchSize, 1, sentMaxLength))
    for i in range(batchSize):
        position[i][0] = np.arange(int(sentLength[i][0][0]))

    return (contxtWords, aspectWords, labels, position, sentLength)

# test_contxtWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, sentMaxLength])
# test_aspectWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, 1])
# test_labels_placeholder      = tf.placeholder(tf.float32, [batchSize, 2])
# test_position_placeholder    = tf.placeholder(tf.float32, [batchSize, 1, sentMaxLength])
# test_sentLength_placeholder  = tf.placeholder(tf.float32, [batchSize, 1, 1])

contxtWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, sentMaxLength])
aspectWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, 1])
labels_placeholder      = tf.placeholder(tf.float32, [batchSize, 2])
position_placeholder    = tf.placeholder(tf.float32, [batchSize, 1, sentMaxLength])
sentLength_placeholder  = tf.placeholder(tf.float32, [batchSize, 1, 1])

attention_W = tf.Variable(np.random.rand(1, 2 * vectorLength), dtype = tf.float32)
attention_b = tf.Variable(np.random.rand(1), dtype = tf.float32)

linearLayer_W = tf.Variable(np.random.rand(vectorLength, vectorLength) / 100, dtype=tf.float32)
linearLayer_b = tf.Variable(np.random.rand(vectorLength, 1) / 100, dtype = tf.float32)

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
print(linearLayerOut)
softOut = tf.nn.softmax(linearLayerOut, 1)
print(softOut)
calssification = tf.reduce_sum(tf.nn.softmax(linearLayerOut), 2)

losses = tf.nn.softmax_cross_entropy_with_logits(calssification, labels_placeholder)
total_loss = tf.reduce_sum(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []

    # contxtWords,aspectWords,labels,position,sentLength = generateData()
    for epoch_idx in range(num_epoches):
        contxtWords,aspectWords,labels,position,sentLength = generateData()

        print("New data, epoch", epoch_idx)

        _calssification, _total_loss, _train_step =  sess.run(
            [softOut, total_loss, train_step],
            feed_dict=
            {
                contxtWords_placeholder:contxtWords,
                aspectWords_placeholder:aspectWords,
                labels_placeholder     :labels,
                position_placeholder   :position,
                sentLength_placeholder :sentLength
            }
        )
        loss_list.append(_total_loss)
        print(_calssification)
        print("Step", epoch_idx, "Loss", _total_loss, "train_step", _train_step)
        