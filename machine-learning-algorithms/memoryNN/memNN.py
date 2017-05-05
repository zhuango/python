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
hopNumber = 2
classNumber = 4
num_epoches = 2000
weightDecay = 0.001

trainDatasetPath = "/home/laboratory/memoryCorpus/train/"
testDatasetPath = "/home/laboratory/memoryCorpus/test/"
resultOutput = '/home/laboratory/memoryCorpus/result/'
if not os.path.exists(resultOutput):
    os.makedirs(resultOutput)

def loadData(datasetPath, shape, sentMaxLength):
    print("load " + datasetPath)
    datasets = np.loadtxt(datasetPath, np.float)
    datasets = np.reshape(datasets, shape)
    return datasets

def shuffleDatasets(datasets, orders):
    shuffleDatasets = np.zeros(datasets.shape)
    index = 0
    for i in orders:
        shuffleDatasets[index] = datasets[i]
        index += 1
    del datasets
    return shuffleDatasets

def generateData(datasetPath,corpusSize, sentMaxLength):
    contxtWordsDir = datasetPath + 'contxtWords'
    aspectWordsDir = datasetPath + 'aspectWords'
    labelsDir      = datasetPath + 'labels'
    positionsDir   = datasetPath + 'positions'
    sentLengthsDir = datasetPath + 'sentLengths'
    maskDir        = datasetPath + 'mask'
    
    contxtWords = loadData(contxtWordsDir, (corpusSize, vectorLength, sentMaxLength), sentMaxLength)
    aspectWords = loadData(aspectWordsDir, (corpusSize, vectorLength, 1), sentMaxLength)
    labels      = loadData(labelsDir, (corpusSize, classNumber, 1), sentMaxLength)
    position    = loadData(positionsDir, (corpusSize, 1, sentMaxLength), sentMaxLength)
    sentLength  = loadData(sentLengthsDir, (corpusSize, 1, 1), sentMaxLength)
    mask        = loadData(maskDir, (corpusSize, 1, sentMaxLength), sentMaxLength)

    return (contxtWords, aspectWords, labels, position, sentLength, mask)

def plot(loss_list):
    plt.cla()
    plt.plot(loss_list)
    plt.draw()
    plt.pause(0.0001)

contxtWords_placeholder = tf.placeholder(tf.float32, [vectorLength, None], name="contxtWords")#
aspectWords_placeholder = tf.placeholder(tf.float32, [vectorLength, 1], name="aspectWords")
labels_placeholder      = tf.placeholder(tf.float32, [classNumber, 1], name="labels")
position_placeholder    = tf.placeholder(tf.float32, [1, None], name="position")# 
sentLength_placeholder  = tf.placeholder(tf.float32, [1, 1], name="sentLength")
mask_placeholder        = tf.placeholder(tf.float32, [1, None], name="mask")

attention_W = tf.Variable(np.random.uniform(-0.01, 0.01, (1, 2 * vectorLength)), dtype = tf.float32, name="attention_W")
attention_b = tf.Variable(np.random.uniform(-0.01, 0.01), dtype = tf.float32, name="attention_b")

linearLayer_W = tf.Variable(np.random.uniform(-0.01, 0.01, (vectorLength, vectorLength)) , dtype=tf.float32, name="linearLayer_W")
linearLayer_b = tf.Variable(np.random.uniform(-0.01, 0.01, (vectorLength, 1)) , dtype = tf.float32, name="linearLayer_b")

softmaxLayer_W = tf.Variable(np.random.uniform(-0.01, 0.01, (classNumber, vectorLength)), dtype= tf.float32, name="softmaxLayer_W")
softmaxLayer_b = tf.Variable(np.random.uniform(-0.01, 0.01, (classNumber, 1)), dtype= tf.float32, name="softmaxLayer_b")

vaspect = aspectWords_placeholder

Vi = 1.0 - position_placeholder / sentLength_placeholder - (hopNumber / vectorLength) * (1.0 - 2.0 * (position_placeholder / sentLength_placeholder))
Mi = Vi * contxtWords_placeholder

for i in range(hopNumber):
    
    expanded_vaspect = vaspect
    for j in range(sentMaxLength - 1):
        expanded_vaspect = tf.concat(1, [expanded_vaspect, vaspect])
    
    attentionInputs = tf.concat(0, [Mi, expanded_vaspect])
    gi = tf.tanh(tf.matmul(attention_W, attentionInputs) + attention_b) + mask_placeholder
    
    alpha = tf.nn.softmax(gi)
    linearLayerOut = tf.matmul(linearLayer_W, vaspect) + linearLayer_b
    vaspect = tf.reduce_sum(alpha * Mi, 1, True) + linearLayerOut

finallinearLayerOut = tf.matmul(softmaxLayer_W, vaspect) + softmaxLayer_b

# regu  = tf.reduce_sum(attention_W * attention_W)
# regu += tf.reduce_sum(attention_b * attention_b)
# regu += tf.reduce_sum(linearLayer_W * linearLayer_W) 
# regu += tf.reduce_sum(linearLayer_b * linearLayer_b)
# regu += tf.reduce_sum(softmaxLayer_W * softmaxLayer_W)
# regu += tf.reduce_sum(softmaxLayer_b * softmaxLayer_b)
# regu = weightDecay * regu
calssification = tf.nn.softmax(finallinearLayerOut - tf.reduce_max(finallinearLayerOut), dim=0)
total_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(finallinearLayerOut - tf.reduce_max(finallinearLayerOut), labels_placeholder, dim=0))

ada = tf.train.AdagradOptimizer(0.1)# 0.3 for hopNumber = 1
train_step = ada.minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []

    contxtWords, aspectWords, labels, position, sentLength,  mask  = generateData(trainDatasetPath, corpusSize, sentMaxLength)
    contxtWordsT,aspectWordsT,labelsT,positionT,sentLengthT, maskT = generateData(testDatasetPath, testDataSize, testMaxLength)
    for epoch_idx in range(num_epoches):
        results = []
        sum_loss= 0.0
        print("New data, epoch", epoch_idx)

        orders = np.arange(corpusSize)
        np.random.shuffle(orders)
        contxtWords = shuffleDatasets(contxtWords, orders)
        aspectWords = shuffleDatasets(aspectWords, orders)
        labels      = shuffleDatasets(labels, orders)
        position    = shuffleDatasets(position, orders)
        sentLength  = shuffleDatasets(sentLength, orders)
        mask        = shuffleDatasets(mask, orders)
        
        count = 0
        correct = 0
        for i in range(corpusSize):
            _calssification, _total_loss, _train_step, _attention_W =  sess.run(
                [calssification, total_loss, train_step, attention_W],
                feed_dict=
                {
                    contxtWords_placeholder:contxtWords[i],
                    aspectWords_placeholder:aspectWords[i],
                    labels_placeholder     :labels[i],
                    position_placeholder   :position[i],
                    sentLength_placeholder :sentLength[i],
                    mask_placeholder       :mask[i]
                }
            )
            sum_loss += _total_loss
            if np.argmax(_calssification.reshape(4)) == np.argmax(labels[i]):
                correct += 1.0
            count += 1
            # print(_attention_W)
            # print(sentLength[i])
        
        print("Iteration", epoch_idx, "Loss", sum_loss / (corpusSize * 2), "train_step", _train_step, "Accuracy: ", float(correct / count))
        loss_list.append(sum_loss / (corpusSize * 2))
        plot(loss_list)
        
        count = 0
        correct = 0
        for i in range(testDataSize):

            _calssification =  sess.run(
                calssification,
                feed_dict=
                {
                    contxtWords_placeholder:contxtWordsT[i],
                    aspectWords_placeholder:aspectWordsT[i],
                    labels_placeholder     :labelsT[i],
                    position_placeholder   :positionT[i],
                    sentLength_placeholder :sentLengthT[i],
                    mask_placeholder       :maskT[i]
                }
            )
            results.append(_calssification.reshape(4))            
            if np.argmax(_calssification.reshape(4)) == np.argmax(labelsT[i]):
                correct += 1.0
            count += 1
        print("test Accuracy: ", float(correct / count))
        np.savetxt(resultOutput + "predict_" + str(epoch_idx) + ".txt", np.asarray(results, dtype=np.float32), fmt='%.5f',delimiter=' ')