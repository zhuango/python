import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

corpusSize = 1977#2358
testDataSize = 49
testMaxLength = 82
batchSize = 1
vectorLength = 50
sentMaxLength = 82
hopNumber = 3
classNumber = 4
num_epoches = 30
weightDecay = 0.001
k = 1

#0.7346938775510204
#0.7551020408163265

trainDatasetPath = "/home/laboratory/memoryCorpus/train/"
testDatasetPath = "/home/laboratory/memoryCorpus/test/"
resultOutput = '/home/laboratory/memoryCorpus/result/'
if not os.path.exists(resultOutput):
    os.makedirs(resultOutput)

def loadData(datasetPath, shape, sentMaxLength, t=np.float):
    print("load " + datasetPath)
    datasets = np.loadtxt(datasetPath, t)
    datasets = np.reshape(datasets, shape)
    return datasets

def shuffleDatasets(datasets, orders, t=np.float):
    shuffleDatasets = np.zeros(datasets.shape, dtype=t)
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
    
    contxtWords = loadData(contxtWordsDir, (corpusSize, sentMaxLength, vectorLength), sentMaxLength)
    aspectWords = loadData(aspectWordsDir, (corpusSize, vectorLength, 1), sentMaxLength)
    labels      = loadData(labelsDir, (corpusSize, classNumber, 1), sentMaxLength, np.int32)
    position    = loadData(positionsDir, (corpusSize, 1, sentMaxLength), sentMaxLength)
    sentLength  = loadData(sentLengthsDir, (corpusSize, 1, 1), sentMaxLength, np.int32)

    return (contxtWords, aspectWords, labels, position, sentLength)

def plot(loss_list):
    plt.cla()
    plt.plot(loss_list)
    plt.draw()
    plt.pause(0.0001)

attention_w  = Variable(torch.FloatTensor(np.random.uniform(-0.1, 0.1, (k, vectorLength))), requires_grad=True)
attention_wg = Variable(torch.FloatTensor(np.random.uniform(-0.1, 0.1, (k, vectorLength))), requires_grad=True)
attention_wh = Variable(torch.FloatTensor(np.random.uniform(-0.1, 0.1, (1, k))), requires_grad=True)

linearLayer_W = Variable(torch.FloatTensor(np.random.uniform(-0.01, 0.01, (vectorLength, vectorLength))), requires_grad=True)
linearLayer_b = Variable(torch.FloatTensor(np.random.uniform(-0.01, 0.01, (vectorLength, 1))), requires_grad=True)

softmaxLayer_W = Variable(torch.FloatTensor(np.random.uniform(-0.01, 0.01, (classNumber, vectorLength))), requires_grad=True)
softmaxLayer_b = Variable(torch.FloatTensor(np.random.uniform(-0.01, 0.01, (classNumber, 1))), requires_grad=True)

softmax = torch.nn.Softmax()

def memModel(contxtWords, aspectWords, position, sentLength):
    vaspect = aspectWords
    for i in range(hopNumber):
        Vi = 1.0 - position / sentLength - (i / vectorLength) * (1.0 - 2.0 * (position / sentLength))
        Mi = Vi.expand_as(contxtWords) * contxtWords

        attentionContxt = torch.mm(attention_w, Mi)
        alphaContxt = softmax(torch.mm(attention_wh, torch.tanh(attentionContxt)))
        g0 = torch.sum(alphaContxt.expand_as(Mi) * Mi, 1)

        attentionE = torch.mm(attention_w, vaspect) + torch.mm(attention_wg, g0).expand(k, 1)
        alphaE = softmax(torch.mm(attention_wh, torch.tanh(attentionE)))
        ge = torch.sum(alphaE.expand_as(vaspect) * vaspect, 1)

        attentionEContxt = torch.mm(attention_w, Mi) + torch.mm(attention_wg, ge).expand(k, sentLength)
        alphaEContxt = softmax(torch.mm(attention_wh, torch.tanh(attentionEContxt)))
        gContxt = torch.sum(alphaEContxt.expand_as(Mi) * Mi, 1)

        linearLayerOut = torch.mm(linearLayer_W, vaspect) + linearLayer_b
        vaspect = gContxt + linearLayerOut

    finallinearLayerOut = torch.mm(softmaxLayer_W, vaspect) + softmaxLayer_b
    return finallinearLayerOut

loss_function = torch.nn.NLLLoss()

def lossModel(contxtWords, aspectWords, position, sentLength, labels):

    finallinearLayerOut = memModel(Variable(torch.Tensor(contxtWords)), 
                                    Variable(torch.Tensor(aspectWords)), 
                                    Variable(torch.Tensor(position)),
                                    sentLength)

    log_prob = F.log_softmax(finallinearLayerOut.view(1, classNumber))

    label = int(labels.argmax())
    total_loss = loss_function(log_prob, Variable(torch.LongTensor([label])))
    calssification = softmax(finallinearLayerOut.view(1, classNumber))
    return total_loss, calssification

def testModel(contxtWords, aspectWords, position, sentLength):

    finallinearLayerOut = memModel(Variable(torch.Tensor(contxtWords)), 
                                    Variable(torch.Tensor(aspectWords)), 
                                    Variable(torch.Tensor(position)),
                                    sentLength)
    calssification = softmax(finallinearLayerOut.view(1, classNumber))

    return calssification

parameters = [attention_w, attention_wg, attention_wh, linearLayer_W, linearLayer_b, softmaxLayer_W, softmaxLayer_b]
optimizer = optim.Adagrad(parameters, lr = 0.05)

loss_list = []

contxtWords, aspectWords, labels, position, sentLength  = generateData(trainDatasetPath, corpusSize, sentMaxLength)
contxtWordsT,aspectWordsT,labelsT,positionT,sentLengthT = generateData(testDatasetPath, testDataSize, testMaxLength)
for epoch_idx in range(num_epoches):
    results = []
    sum_loss= 0.0
    print("New data, epoch", epoch_idx)

    orders = np.arange(corpusSize)
    np.random.shuffle(orders)
    contxtWords = shuffleDatasets(contxtWords, orders)
    aspectWords = shuffleDatasets(aspectWords, orders)
    labels      = shuffleDatasets(labels, orders, np.int64)
    position    = shuffleDatasets(position, orders)
    sentLength  = shuffleDatasets(sentLength, orders, np.int64)
        
    count = 0
    correct = 0
    for i in range(corpusSize):
        total_loss,  calssification = lossModel(contxtWords[i,0:sentLength[i]].T, aspectWords[i], position[i,:,0:sentLength[i]], int(sentLength[i]), labels[i])

        total_loss.backward()
        optimizer.step()

        for para in parameters:
            para.grad.data.zero_()
        sum_loss += total_loss

        if np.argmax(calssification.data.numpy()) == np.argmax(labels[i]):
            correct += 1.0
        count += 1
        
    print("Iteration", epoch_idx, "Loss", sum_loss.data.numpy() / (corpusSize * 2), "Accuracy: ", float(correct / count))
        
    count = 0
    correct = 0
    for i in range(testDataSize):
        calssification = testModel(contxtWordsT[i,0:sentLengthT[i]].T, aspectWordsT[i], positionT[i,:,0:sentLengthT[i]], int(sentLengthT[i]))
     
        if np.argmax(calssification.data.numpy()) == np.argmax(labelsT[i]):
            correct += 1.0
        count += 1
    print("test Accuracy: ", float(correct / count))
    #np.savetxt(resultOutput + "predict_" + str(epoch_idx) + ".txt", np.asarray(results, dtype=np.float32), fmt='%.5f',delimiter=' ')