# /usr/bin/python3

import numpy as np
from collections import OrderedDict
import math

class Sample():
    LabelTable = OrderedDict()
    WordsTable = OrderedDict()
    DictLength = 0
    LabelSize  = 0
    def __init__(self, s, labels=None):
        self.sequence   = list(s)
        self.Labels     = labels
        self.Length     = len(s)
        self.unigram    = {}
        self.bigram     = {}

        # LabelTable = Sample.LabelTable
        WordsTable = Sample.WordsTable
        for word in self.sequence:
            if word not in WordsTable:
                WordsTable[word]  = Sample.DictLength
                Sample.DictLength += 1

    def hashKey(self, seqNum, y0State, y1State = None):
        if y2State != None:
            return str(seqNum) + "_" + str(y0State) + "+" + str(y1State)
        else:
            return str(seqNum) + "_" + str(y0State)
            
    def GetFeature(self, seqNum, y0State, y1State = None):
        WordsTable = Sample.WordsTable
        word = self.sequence[seqNum]
        if y1State != None:
            #return self.bigram [self.sequence[seqNum]][y0State][y1State]
            return WordsTable[word] * (Sample.LabelSize ** 2) + y0State * Sample.LabelSize + y1State
        else:
            return WordsTable[word] * Sample.LabelSize + y0State

class CRFBin:
    def __init__(self, nodeFeatureSize, edgeFeatureSize, labelStateSize):
        # self.mWnode = np.random.uniform(0, 1, nodeFeatureSize)
        # self.mWedge = np.random.uniform(0, 1, edgeFeatureSize)
        self.mWnode = np.zeros(nodeFeatureSize)
        self.mWedge = np.zeros(edgeFeatureSize)
        self.mLabelStateSize = labelStateSize

    def LogPotentialTable(self, sequence):
        seqLength = sequence.Length
        logPotential0 = np.zeros(self.mLabelStateSize)
        logPotentials = np.zeros((seqLength-1, self.mLabelStateSize, self.mLabelStateSize))

        for i in range(0, self.mLabelStateSize):
            logPotential0[i] = self.mWnode[sequence.GetFeature(0, i)].sum()
        for i in range(1, seqLength):
            for j in range(0, self.mLabelStateSize):
                for k in range(0, self.mLabelStateSize):
                    node = self.mWnode[sequence.GetFeature(i, k)].sum()
                    edge = self.mWedge[sequence.GetFeature(i, j, k)].sum()
                    logPotentials[i-1][j][k] = node + edge
        return (logPotential0, logPotentials)
    
    def forward(self, logPotential0, logPotentials, seqLength):
        messages = np.zeros((seqLength, self.mLabelStateSize))
        messages[0] = logPotential0
        for i in range(1, seqLength):
            Fs = logPotentials[i-1] + messages[i-1].reshape(self.mLabelStateSize, 1)
            b = Fs.max(0)
            integralPotentials = np.exp(Fs - b).sum(0)
            messages[i] = np.log(integralPotentials) + b
        return messages

    def backward(self, logPotentials, seqLength):
        messages = np.zeros((seqLength, self.mLabelStateSize))
        for i in reversed(range(0, seqLength-1)):
            Fs = logPotentials[i] + messages[i+1]
            b = Fs.max(1, keepdims=True)
            integralPotentials = np.exp(Fs - b).sum(1)
            messages[i] = np.log(integralPotentials) + b.T
        return messages

    def LogLikelihood(self, sequence):
        seqLength = sequence.Length
        seqStates = sequence.Labels
        (logPotential0, logPotentials) = self.LogPotentialTable(sequence)
        messages = self.forward(logPotential0, logPotentials, seqLength)
        
        b = messages[seqLength-1].max()
        logNormalized = np.log(np.exp(messages[seqLength-1] - b).sum()) + b

        logLikelihood = logPotential0[seqStates[0]]
        for i in range(1, seqLength):
            y1 = seqStates[i-1]
            y2 = seqStates[i]
            logLikelihood += logPotentials[i-1][y1][y2]
        logLikelihood -= logNormalized
        return logLikelihood

    def gradientOfNormalizedRespectW(self, sequence):
        seqLength  = sequence.Length
        
        (logPotential0, logPotentials) = self.LogPotentialTable(sequence)
        #print(logPotentials)
        forwardMessages  = self.forward(logPotential0, logPotentials, seqLength)
        backwardMessages = self.backward(logPotentials, seqLength)

        b = forwardMessages[seqLength-1].max()
        logNormalized = np.log(np.exp(forwardMessages[seqLength-1] - b).sum()) + b

        WnodeGradient = np.zeros_like(self.mWnode)
        WedgeGradient = np.zeros_like(self.mWedge)

        for i in range(0, seqLength):
            for j in range(0, self.mLabelStateSize):
                #print(forwardMessages[i][j] + backwardMessages[i][j] - logNormalized)
                WnodeGradient[sequence.GetFeature(i, j)] += np.exp(forwardMessages[i][j] + backwardMessages[i][j] - logNormalized).clip(0.0, 1.0)
        
        for i in range(1, seqLength):
            for j in range(0, self.mLabelStateSize):
                for k in range(0, self.mLabelStateSize):
                    #print(forwardMessages[i-1][j] + logPotentials[i-1][j][k] + backwardMessages[i][k] - logNormalized)
                    WedgeGradient[sequence.GetFeature(i, j, k)] += np.exp(forwardMessages[i-1][j] + logPotentials[i-1][j][k] + backwardMessages[i][k] - logNormalized).clip(0.0, 1.0)

        # print(WnodeGradient, WedgeGradient)
        return (WnodeGradient, WedgeGradient)

    def Sample(self, sequence):
        seqLength = sequence.Length

        (logPotential0, logPotentials) = self.LogPotentialTable(sequence)
        labels = np.zeros(seqLength, dtype=np.int32)

        pathMatrix = np.zeros((seqLength, self.mLabelStateSize))
        w = logPotential0
        for i in range(1, seqLength):
            w = logPotentials[i - 1]+ np.reshape(w, (self.mLabelStateSize, 1))
            pathMatrix[i-1] = w.argmax(0)
            w = w.max(0)
        labels[-1] = w.argmax()
        for i in reversed(range(0, seqLength - 1)):
            labels[i] = pathMatrix[i][labels[i + 1]]
        return labels

    def SGA(self, sequences ,iterations=20, a0=1, validate=None):
        # rate = 5
        oldLikelihood = -10000000000000000
        earlyStopCount = 3
        dataSize = len(sequences)
        for iteration in range(0, iterations):
            rate = a0 / (math.sqrt(iteration) + 1)
            print(rate)
            print("Iteration: " + str(iteration))

            try:
                #np.random.shuffle(sequences)
                for i in range(0, dataSize):
                    sequence   = sequences[i]
                    labels     = sequence.Labels
                    #print("cal gradient of normalize respect W ...")
                    (WnodeGradient, WedgeGradient) = self.gradientOfNormalizedRespectW(sequence)
                    self.mWnode -= WnodeGradient * rate
                    self.mWedge -= WedgeGradient * rate

                    for j in range(0, sequence.Length):
                        self.mWnode[sequence.GetFeature(j, labels[j])] += rate

                    for j in range(1, sequence.Length):
                        self.mWedge[sequence.GetFeature(j, labels[j-1], labels[j])] += rate
                    # self.mWnode -= self.mWnode * rate * 0.1
                    # self.mWedge -= self.mWedge * rate * 0.1

                print("cal loglikehood ...")
                likelihood = 0.0
                for i in range(0, dataSize):
                    sequence = sequences[i]
                    currentLikelihood = self.LogLikelihood(sequence)
                    likelihood += currentLikelihood
                    #print(currentLikelihood)
                print("Loglihood: " + str(float(likelihood) / dataSize))
                if likelihood <= oldLikelihood:
                    earlyStopCount -= 1
                    if earlyStopCount == 0:
                        return
                else:
                    earlyStopCount = 3
                oldLikelihood = likelihood
            except:
                return 