import numpy as np
import tensorflow as tf
import os

corpusSize = 2358 
batchSize = 2358
vectorLength = 50
sentMaxLength = 82
hopNumber = 1
classNumber = 4
num_epoches = 1000
resultOutput = '/home/jason/memoryCorpus/result/'
if not os.path.exists(resultOutput):
    os.makedirs(resultOutput)
def generateData():
    contxtWordsDir = '/home/jason/memoryCorpus/contxtWords'
    aspectWordsDir = '/home/jason/memoryCorpus/aspectWords'
    labelsDir      = '/home/jason/memoryCorpus/labels'
    positionsDir   = '/home/jason/memoryCorpus/positions'
    sentLengthsDir = '/home/jason/memoryCorpus/sentLengths'

    print("load context words vector...")
    contxtWords = np.loadtxt(contxtWordsDir, np.float).reshape(batchSize, vectorLength, sentMaxLength)
    print("load aspect words vector...")
    aspectWords = np.loadtxt(aspectWordsDir, np.float).reshape(batchSize, vectorLength, 1)
    print("load labels...")
    labels      = np.loadtxt(labelsDir, np.float).reshape(batchSize, classNumber)
    print("load position...")
    position    = np.loadtxt(positionsDir, np.float).reshape(batchSize, 1, sentMaxLength)
    print("load sentLength...")
    sentLength  = np.loadtxt(sentLengthsDir, np.float).reshape(batchSize, 1, 1)

    return (contxtWords, aspectWords, labels, position, sentLength)
def generateDataFake():
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

contxtWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, sentMaxLength], name="contxtWords")
aspectWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, 1], name="aspectWords")
labels_placeholder      = tf.placeholder(tf.float32, [batchSize, classNumber], name="labels")
position_placeholder    = tf.placeholder(tf.float32, [batchSize, 1, sentMaxLength], name="position")
sentLength_placeholder  = tf.placeholder(tf.float32, [batchSize, 1, 1], name="sentLength")

attention_W = tf.Variable(np.random.uniform(-0.01, 0.01, (1, 2 * vectorLength)), dtype = tf.float32, name="attention_W")
attention_b = tf.Variable(np.random.uniform(-0.01, 0.01), dtype = tf.float32, name="attention_b")

linearLayer_W = tf.Variable(np.random.uniform(-0.01, 0.01, (vectorLength, vectorLength)) , dtype=tf.float32, name="linearLayer_W")
linearLayer_b = tf.Variable(np.random.uniform(-0.01, 0.01, (vectorLength, 1)) , dtype = tf.float32, name="linearLayer_b")

softmaxLayer_W = tf.Variable(np.random.uniform(-0.01, 0.01, (classNumber, vectorLength)), dtype= tf.float32, name="softmaxLayer_W")
softmaxLayer_b = tf.Variable(np.random.uniform(-0.01, 0.01, (classNumber, 1)), dtype= tf.float32, name="softmaxLayer_b")

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
calssification = tf.reshape(tf.nn.softmax(linearLayerOut, 1), [batchSize, classNumber])

losses = tf.nn.softmax_cross_entropy_with_logits(calssification, labels_placeholder)
total_loss = tf.reduce_sum(losses)

#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(total_loss)
train_step = tf.train.AdagradOptimizer(0.15).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []

    #merged_summary_op = tf.merge_all_summaries()
    
    contxtWords,aspectWords,labels,position,sentLength = generateData()
    for epoch_idx in range(num_epoches):
        #contxtWords,aspectWords,labels,position,sentLength = generateData()

        print("New data, epoch", epoch_idx)

        _calssification, _total_loss, _train_step =  sess.run(
            [calssification, total_loss, train_step],
            feed_dict=
            {
                contxtWords_placeholder:contxtWords,
                aspectWords_placeholder:aspectWords,
                labels_placeholder     :labels,
                position_placeholder   :position,
                sentLength_placeholder :sentLength
            }
        )
        #summary_writer = tf.train.SummaryWriter('/tmp/aspect_logs', sess.graph)
        #summary_writer.add_summary(_calssification, epoch_idx)
        loss_list.append(_total_loss)
        np.savetxt(resultOutput + "predict_" + str(epoch_idx) + ".txt", _calssification, fmt='%.5f',delimiter=' ')
        print("Step", epoch_idx, "Loss", _total_loss, "train_step", _train_step)