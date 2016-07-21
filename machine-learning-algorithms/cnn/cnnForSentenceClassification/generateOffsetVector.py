# /usr/bin/python2.7
import numpy
import math
import os
def getMaxMinOffset(offsetFileName):
    maxOffset = 0
    minOffset = 0
    with open(offsetFileName, 'r') as f:
        for line in f:
            wordOffsets = line.strip().split(" ")
            for wordOffsetCouple in wordOffsets:
                offsetCouple = wordOffsetCouple.split("&")
                for offset in offsetCouple:
                    if int(offset) > maxOffset:
                        print(offset)
                        maxOffset = int(offset)
                    elif int(offset) < minOffset:
                        minOffset = int(offset)
    return (maxOffset, minOffset)

def generateOffsetVector(offsetVectorFileName, minOffset, maxOffset, vectorLength):
    with open(offsetVectorFileName, 'w') as f:
        a = minOffset
        while a <= maxOffset:
            vectorStr = ""
            for value in numpy.random.uniform(-0.25,0.25,vectorLength):
                vectorStr += str(value) + " "
            f.write(str(a) + " " + vectorStr.strip() + "\n")
            a += 1
def generateOffsetVector(wordVectorFile, offsetName, contextName, wordVectorLength, vectorLength, preffix):
    word_dict={}
    word_dict_withOffset = {}
    offset_dict = {}
    with open(wordVectorFile, "r") as f:
        for line in f:
            strs =line.strip().split(' ')
            word_dict[str.lower(strs[0])] = line[len(strs[0]) + 1:].strip()
    with open(offsetName, "r") as f:
        context = open(contextName, "r")
        newcontext = open(contextName + ".withOffset", 'w')
        for line in f:
            newsent = ""
            sent = context.readline().strip()
            words = sent.split(" ")
            # print(line)
            # print(sent)
            # print("################################")
            i = 0
            for offset in line.strip().split(" "):
                wordStr = words[i] + offset
                newsent += wordStr + " "
                vectorStr = ""
                if words[i] not in word_dict:
                    for value in numpy.random.uniform(-0.25, 0.25, wordVectorLength):
                        vectorStr += str(value) + " "
                    vectorStr = vectorStr.strip()
                else:
                    vectorStr = word_dict[words[i]]
                if offset not in offset_dict:
                    offsetvectorStr = ""
                    for value in numpy.random.uniform(-0.25,0.25,vectorLength * 2):
                         offsetvectorStr += str(value) + " "
                    offset_dict[offset] = offsetvectorStr.strip()
                vectorStr += " " + offset_dict[offset]
                word_dict_withOffset[wordStr] = vectorStr
                i+=1
        
            newcontext.write(newsent.strip() + "\n")
        context.close()
    
    newVectorFileName = wordVectorFile + preffix+".extent";
    newVectorFile = open(newVectorFileName, "w")
    for word in word_dict_withOffset:
        newVectorFile.write(word + " " + word_dict_withOffset[word] + "\n")
    newVectorFile.close()
    return newVectorFileName

def mergeVector(vectorFileA, vectorFileB):
    word_dict = {}
    newVectorFile = "total.vector"
    with open(vectorFileA, "r") as f:
        for line in f:
            strs =line.strip().split(' ')
            word_dict[str.lower(strs[0])] = line[len(strs[0]) + 1:].strip()
    with open(vectorFileB, 'r') as f:
        for line in f:
            strs =line.strip().split(' ')
            word_dict[str.lower(strs[0])] = line[len(strs[0]) + 1:].strip()
    with open(newVectorFile, "w") as f:
        for word in word_dict:
            f.write(word + " " + word_dict[word] + "\n")
    return newVectorFile

if __name__ == "__main__":
    rootpath = "/home/laboratory/corpusYang/"
    traiContext = "/home/laboratory/corpusYang/train_word.cnn"
    testContext = "/home/laboratory/corpusYang/test_word.cnn"

    testOffset = "/home/laboratory/corpusYang/numberTestfile"
    traiOffset = "/home/laboratory/corpusYang/numberTrainfile"
    wordVectorFile = "/home/laboratory/corpusYang/finalWordEmbeddings_50dim.txt"

    newVectorFileNametrain = generateOffsetVector(wordVectorFile, traiOffset, traiContext, 10, "train")
    newVectorFileNametest = generateOffsetVector(wordVectorFile, testOffset, testContext, 10, "test")
    # max, min = getMaxMinOffset(testOffset)
    # generateOffsetVector(testOffset + ".vector", min,max ,5)

    # max, min = getMaxMinOffset(traiOffset)
    # generateOffsetVector(traiOffset + ".vector", min,max ,5)
    mergeVector("embedding100.vectest.extent", "embedding100.vectrain.extent")