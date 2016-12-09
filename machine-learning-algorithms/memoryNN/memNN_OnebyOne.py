import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

corpusSize = 1977#2358
testDataSize = 49
testMaxLength = 82
batchSize = 1
vectorLength = 50
sentMaxLength = 82
hopNumber = 1
classNumber = 4
num_epoches = 2000
weightDecay = 0.01

trainDatasetPath = "/home/laboratory/memoryCorpus/train/"
testDatasetPath = "/home/laboratory/memoryCorpus/test/"
resultOutput = '/home/laboratory/memoryCorpus/result/'
if not os.path.exists(resultOutput):
    os.makedirs(resultOutput)

def loadData(datasetPath, shape, sentMaxLength, orders):
    print("load " + datasetPath)
    datasets = np.loadtxt(datasetPath, np.float)
    datasets = np.reshape(datasets, shape)
    shuffleDatasets = np.zeros(shape)
    index = 0
    for i in orders:
        shuffleDatasets[index] = datasets[i]
        index += 1
    del datasets
    return shuffleDatasets
    
def generateData(datasetPath,corpusSize, sentMaxLength):
    batchSizeOffset = batchSize - corpusSize % batchSize
    contxtWordsDir = datasetPath + 'contxtWords'
    aspectWordsDir = datasetPath + 'aspectWords'
    labelsDir      = datasetPath + 'labels'
    positionsDir   = datasetPath + 'positions'
    sentLengthsDir = datasetPath + 'sentLengths'
    maskDir        = datasetPath + 'mask'
    
    orders = np.arange(corpusSize)
    np.random.shuffle(orders)

    contxtWords = loadData(contxtWordsDir, (corpusSize, vectorLength, sentMaxLength), sentMaxLength, orders)
    #contxtWords = np.concatenate( (contxtWords, np.zeros((batchSizeOffset, vectorLength, sentMaxLength))) )
    # print(contxtWords[0])
    
    aspectWords = loadData(aspectWordsDir, (corpusSize, vectorLength, 1), sentMaxLength, orders)
    #aspectWords = np.concatenate( (aspectWords, np.zeros((batchSizeOffset, vectorLength, 1))) )

    labels      = loadData(labelsDir, (corpusSize, classNumber, 1), sentMaxLength, orders)
    #labels      maskDir= np.concatenate((labels, np.zeros((batchSizeOffset, classNumber, 1))))
    
    position    = loadData(positionsDir, (corpusSize, 1, sentMaxLength), sentMaxLength, orders)
    #position    = np.concatenate( (position, np.zeros((batchSizeOffset, 1, sentMaxLength))) )
    
    sentLength  = loadData(sentLengthsDir, (corpusSize, 1, 1), sentMaxLength, orders)
    #sentLength  = np.concatenate( (sentLength, np.zeros((batchSizeOffset, 1, 1)) + sentLength[corpusSize - 1][0][0]) )
    
    mask        = loadData(maskDir, (corpusSize, 1, sentMaxLength), sentMaxLength, orders)

    return (contxtWords, aspectWords, labels, position, sentLength, mask)
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
def plot(loss_list):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    plt.draw()
    plt.pause(0.0001)

# test_contxtWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, sentMaxLength])
# test_aspectWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, 1])
# test_labels_placeholder      = tf.placeholder(tf.float32, [batchSize, 2])
# test_position_placeholder    = tf.placeholder(tf.float32, [batchSize, 1, sentMaxLength])
# test_sentLength_placeholder  = tf.placeholder(tf.float32, [batchSize, 1, 1])

contxtWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, None], name="contxtWords")#
aspectWords_placeholder = tf.placeholder(tf.float32, [batchSize, vectorLength, 1], name="aspectWords")
labels_placeholder      = tf.placeholder(tf.float32, [batchSize, classNumber, 1], name="labels")
position_placeholder    = tf.placeholder(tf.float32, [batchSize, 1, None], name="position")# 
sentLength_placeholder  = tf.placeholder(tf.float32, [batchSize, 1, 1], name="sentLength")
mask_placeholder        = tf.placeholder(tf.float32, [batchSize, 1, None], name="mask")

attention_W = tf.Variable(np.random.uniform(-0.01, 0.01, (1, 2 * vectorLength)), dtype = tf.float32, name="attention_W")
attention_b = tf.Variable(np.random.uniform(-0.01, 0.01), dtype = tf.float32, name="attention_b")

linearLayer_W = tf.Variable(np.random.uniform(-0.01, 0.01, (vectorLength, vectorLength)) , dtype=tf.float32, name="linearLayer_W")
linearLayer_b = tf.Variable(np.random.uniform(-0.01, 0.01, (vectorLength, 1)) , dtype = tf.float32, name="linearLayer_b")

softmaxLayer_W = tf.Variable(np.random.uniform(-0.01, 0.01, (classNumber, vectorLength)), dtype= tf.float32, name="softmaxLayer_W")
softmaxLayer_b = tf.Variable(np.random.uniform(-0.01, 0.01, (classNumber, 1)), dtype= tf.float32, name="softmaxLayer_b")

vaspect = aspectWords_placeholder

for i in range(hopNumber):
    Vi = 1.0 - position_placeholder / sentLength_placeholder - (hopNumber / vectorLength) * (1.0 - 2.0 * (position_placeholder / sentLength_placeholder))
    Mi = Vi * contxtWords_placeholder
    expanded_vaspect = vaspect
    for j in range(sentMaxLength - 1):
        expanded_vaspect = tf.concat(2, [expanded_vaspect, vaspect])
    print(Mi)
    print(expanded_vaspect)
    attentionInputs = tf.unpack(tf.concat(1, [Mi, expanded_vaspect]), axis=0)
    print(attentionInputs)
    gi = tf.pack([tf.tanh(tf.matmul(attention_W, input) + attention_b) for input in attentionInputs]) + mask_placeholder
    
    alpha = tf.nn.softmax(gi)
    linearLayerOut = tf.pack([tf.matmul(linearLayer_W, input) + linearLayer_b for input in tf.unpack(vaspect)])
    vaspect = tf.reduce_sum(alpha * Mi, 2, True) + linearLayerOut

labelsSeries  = tf.unpack(labels_placeholder)
linearLayerOutSeries = [tf.matmul(softmaxLayer_W, input) + softmaxLayer_b for input in tf.unpack(vaspect)]
#layerOut = linearLayerOutSeries[0]
#test = layerOut / tf.reduce_sum(tf.exp(layerOut - tf.reduce_max(layerOut)))
#print(test)
regu  = tf.reduce_sum(attention_W * attention_W)
regu += tf.reduce_sum(attention_b * attention_b)
regu += tf.reduce_sum(linearLayer_W * linearLayer_W) 
regu += tf.reduce_sum(linearLayer_b * linearLayer_b)
regu += tf.reduce_sum(softmaxLayer_W * softmaxLayer_W)
regu = weightDecay * regu

losses = [tf.nn.softmax_cross_entropy_with_logits(layerOut - tf.reduce_max(layerOut), label, dim=0) for layerOut, label in zip(linearLayerOutSeries, labelsSeries)]
total_loss = tf.reduce_sum(losses) + regu

#train_step = tf.train.GradientDescentOptimizer(0.02).minimize(total_loss)

train_step = tf.train.AdagradOptimizer(0.01).minimize(total_loss)
calssification = tf.reshape([tf.nn.softmax(layerOut - tf.reduce_max(layerOut), dim=0) for layerOut in linearLayerOutSeries], [batchSize, classNumber])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []

    merged_summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('/tmp/aspect_logs', sess.graph)

    contxtWords, aspectWords, labels, position, sentLength,  mask  = generateData(trainDatasetPath, corpusSize, sentMaxLength)
    contxtWordsT,aspectWordsT,labelsT,positionT,sentLengthT, maskT = generateData(testDatasetPath, testDataSize, testMaxLength)
    for epoch_idx in range(num_epoches):
        #contxtWords,aspectWords,labels,position,sentLength = generateData()
        results = []
        sum_loss= 0.0
        print("New data, epoch", epoch_idx)
        for i in    range(corpusSize // batchSize):
            
            _calssification, _total_loss, _train_step =  sess.run(
                [calssification, total_loss, train_step],
                feed_dict=
                {
                    contxtWords_placeholder:contxtWords[i * batchSize:(i + 1) *batchSize],
                    aspectWords_placeholder:aspectWords[i * batchSize:(i + 1) *batchSize],
                    labels_placeholder     :labels[i * batchSize:(i + 1) *batchSize],
                    position_placeholder   :position[i * batchSize:(i + 1) *batchSize],
                    sentLength_placeholder :sentLength[i * batchSize:(i + 1) *batchSize],
                    mask_placeholder       :mask[i * batchSize:(i + 1) *batchSize]
                }
            )
            sum_loss += _total_loss
            #print(_calssification)
        for i in range(testDataSize):

            _calssification =  sess.run(
                calssification,
                feed_dict=
                {
                    contxtWords_placeholder:contxtWordsT[i * batchSize:(i + 1) *batchSize],
                    aspectWords_placeholder:aspectWordsT[i * batchSize:(i + 1) *batchSize],
                    labels_placeholder     :labelsT[i * batchSize:(i + 1) *batchSize],
                    position_placeholder   :positionT[i * batchSize:(i + 1) *batchSize],
                    sentLength_placeholder :sentLengthT[i * batchSize:(i + 1) *batchSize],
                    mask_placeholder       :maskT[i * batchSize:(i + 1) *batchSize]
                }
            )
            results.append(_calssification.reshape(4))
        np.savetxt(resultOutput + "predict_" + str(epoch_idx) + ".txt", np.asarray(results, dtype=np.float32), fmt='%.5f',delimiter=' ')
        print("Iteration", epoch_idx, "Loss", sum_loss / (corpusSize * 2), "train_step", _train_step)
        loss_list.append(sum_loss)
        #summary_writer.add_summary(_total_loss, epoch_idx)