# /usr/bin/python3

import numpy as np
from collections import defaultdict
import math

class Sample(object):
    LabelTable = {}
    WordsTable = {}
    DictLength = 0
    def __init__(self, s, labels=None):
        self.sequence = list(s)
        self.Labels = labels
        self.Length = len(s)
        self.featureTable = {}
    def hashKey(self, seqNum, y1State, y2State = None):
        if y2State != None:
            return str(seqNum) + "_" + str(y1State) + "+" + str(y2State)
        else:
            return str(seqNum) + "_" + str(y1State)
            
    def GetFeature(self, seqNum, y1State, y2State = None, train = None):
        """ Create the vector of active indicies for this label setting. """
        # label-label-token; bigram features
        #for f in self.sequence[t].attributes:
        #    yield '[%s,%s,%s]' % (yp,y,f)
        # label-token; emission features
        WordsTable = self.WordsTable
        word0  = self.sequence[seqNum]
        if word0 not in WordsTable:
            WordsTable[word0] = np.random.uniform(-1.0, 1.0, 50)
        label0 = self.LabelTable[y1State]

        if y2State != None:
            word1  = self.sequence[seqNum-1]
            if word1 not in WordsTable:
                WordsTable[word1] = np.random.uniform(-1.0, 1.0, 50)
            feature = ((WordsTable[word0] + WordsTable[word1])/2)#.clip(0.0,1.0)
            return feature
        else:
            feature = WordsTable[word0]#.clip(0.0,1.0)
            return feature            

class CRF:
    def __init__(self, maxLen, nodeFeatureSize, edgeFeatureSize, labelStateSize):
        # self.mWnode = np.random.uniform(0, 1, nodeFeatureSize)
        # self.mWedge = np.random.uniform(0, 1, edgeFeatureSize)
        self.mWnode = np.zeros(nodeFeatureSize)
        self.mWedge = np.zeros(edgeFeatureSize)
        self.mLabelStateSize = labelStateSize

    def LogPotentialTable(self, sequence):
        seqLength = sequence.Length
        logPotential0 = np.zeros(self.mLabelStateSize)
        logPotentials = np.zeros((seqLength-1, self.mLabelStateSize, self.mLabelStateSize))

        for i in xrange(0, self.mLabelStateSize):
            logPotential0[i] = np.dot(self.mWnode, sequence.GetFeature(0, i))
        for i in xrange(1, seqLength):
            for j in xrange(0, self.mLabelStateSize):
                for k in xrange(0, self.mLabelStateSize):
                    node = np.dot(self.mWnode, sequence.GetFeature(i, j))
                    edge = np.dot(self.mWedge, sequence.GetFeature(i, j, k))
                    logPotentials[i-1][j][k] = node + edge
        return (logPotential0, logPotentials)
    
    def forward(self, logPotential0, logPotentials, seqLength):
        messages = np.zeros((seqLength, self.mLabelStateSize))
        messages[0] = logPotential0
        for i in xrange(1, seqLength):
            b = logPotentials[i-1].max(0)
            integralPotentials = np.exp(logPotentials[i-1] - b).sum(0)
            messages[i] = np.log(integralPotentials) + b + messages[i-1]
        return messages

    def backward(self, logPotentials, seqLength):
        messages = np.zeros((seqLength, self.mLabelStateSize))
        for i in reversed(xrange(0, seqLength-1)):
            b = logPotentials[i].max(1, keepdims=True)
            integralPotentials = np.exp(logPotentials[i] - b).sum(1)
            messages[i] = np.log(integralPotentials) + b.T + messages[i+1]
        return messages

    def LogLikelihood(self, sequence):
        seqLength = sequence.Length
        seqStates = sequence.Labels
        (logPotential0, logPotentials) = self.LogPotentialTable(sequence)
        messages = self.forward(logPotential0, logPotentials, seqLength)
        
        b = messages[seqLength-1].max()
        logNormalized = np.log(np.exp(messages[seqLength-1] - b).sum()) + b

        logLikelihood = logPotential0[seqStates[0]]
        for i in xrange(1, seqLength):
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

        for i in xrange(0, seqLength):
            for j in xrange(0, self.mLabelStateSize):
                WnodeGradient += np.exp(forwardMessages[i][j] + backwardMessages[i][j] - logNormalized).clip(0.0, 1.0) * sequence.GetFeature(i, j)
        
        for i in xrange(1, seqLength):
            for j in xrange(0, self.mLabelStateSize):
                for k in xrange(0, self.mLabelStateSize):
                    WedgeGradient += np.exp(forwardMessages[i-1][j] + logPotentials[i-1][j][k] + backwardMessages[i][k] - logNormalized).clip(0.0, 1.0) * sequence.GetFeature(i, j, k)

        # print(WnodeGradient, WedgeGradient)
        return (WnodeGradient, WedgeGradient)

    def Sample(self, sequence):
        seqLength = sequence.Length

        (logPotential0, logPotentials) = self.LogPotentialTable(sequence)
        labels = np.zeros(seqLength, dtype=np.int32)

        pathMatrix = np.zeros((seqLength, self.mLabelStateSize))
        w = logPotential0
        for i in xrange(1, seqLength):
            w = logPotentials[i - 1]+ np.reshape(w, (self.mLabelStateSize, 1))
            pathMatrix[i] = w.argmax(0)
            w = w.max(0)
        labels[-1] = w.argmax()
        for i in reversed(xrange(0, seqLength - 1)):
            labels[i] = pathMatrix[i][labels[i + 1]]
        return labels

    def SGA(self, sequences ,iterations=20, a0=1, validate=None):
        # rate = 5
        for iteration in xrange(0, iterations):
            rate = a0 / (math.sqrt(iteration) + 1)
            print(rate)
            print("Iteration: " + str(iteration))
            oldLikelihood = -10000000000000000

            np.random.shuffle(sequences)
            for i in xrange(0, len(sequences)):
                sequence   = sequences[i]
                labels     = sequence.Labels
                (WnodeGradient, WedgeGradient) = self.gradientOfNormalizedRespectW(sequence)
                self.mWnode -= WnodeGradient * rate
                self.mWedge -= WedgeGradient * rate

                for j in xrange(0, sequence.Length):
                    self.mWnode += sequence.GetFeature(j, labels[j]) * rate

                for j in xrange(1, sequence.Length):
                    self.mWedge += sequence.GetFeature(j, labels[j-1], labels[j]) * rate
                # self.mWnode -= self.mWnode * rate * 0.1
                # self.mWedge -= self.mWedge * rate * 0.1
            likelihood = 0.0
            for i in xrange(0, len(sequences)):
                sequence = sequences[i]
                currentLikelihood = self.LogLikelihood(sequence)
                likelihood += currentLikelihood
                print(currentLikelihood)
            print("Loglihood: " + str(likelihood))
            if likelihood < oldLikelihood:
                return
            oldLikelihood = likelihood
            