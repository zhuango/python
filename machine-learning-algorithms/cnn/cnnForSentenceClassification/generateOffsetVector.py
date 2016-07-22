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
                        #print(offset)
                        maxOffset = int(offset)
                    elif int(offset) < minOffset:
                        minOffset = int(offset)
    return (maxOffset, minOffset)

def generateOffsetVectorDict(minOffset, maxOffset, vectorLength):
    word_dict = {}
    a = minOffset
    while a <= maxOffset:
        vectorStr = ""
        for value in numpy.random.uniform(-0.25,0.25,vectorLength):
            vectorStr += str(value) + " "
        word_dict[str(a)] = vectorStr.strip()
        a += 1
    return word_dict
def generateWordVectorDict(wordVectorFile):
    word_dict = {}
    with open(wordVectorFile, "r") as f:
        for line in f:
            strs =line.strip().split(' ')
            word_dict[str.lower(strs[0])] = line[len(strs[0]) + 1:].strip()
    return word_dict
def generateOffsetVector(word_dict, offset_dict, offsetName, contextName, wordVectorLength, vectorLength, preffix):

    word_dict_withOffset = {}
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
                offsets = offset.split("&")
                wordStr = words[i] + offset
                newsent += wordStr + " "
                vectorStr = ""
                if words[i] not in word_dict:
                    for value in numpy.random.uniform(-0.25, 0.25, wordVectorLength):
                        vectorStr += str(value) + " "
                    vectorStr = vectorStr.strip()
                else:
                    vectorStr = word_dict[words[i]]
                # for offs in offsets:
                #     if offs not in offset_dict:
                #         offsetvectorStr = ""
                #         for value in numpy.random.uniform(-0.25,0.25,vectorLength):
                #             offsetvectorStr += str(value) + " "
                #         offset_dict[offs] = offsetvectorStr.strip()
                vectorStr += " " + offset_dict[offsets[0]] + " " + offset_dict[offsets[1]]
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

def mergeVector(vectorFileA, vectorFileB, preffix):
    word_dict = {}
    newVectorFile = preffix + ".vector"
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
    rootpath = "G:/liuzhuang/corpusYang/"
    traiContext = "G:/liuzhuang/corpusYang/Word_Seq_win3/train_word.cnn"
    testContext = "G:/liuzhuang/corpusYang/Word_Seq_win3/test_word.cnn"

    testOffset = "G:/liuzhuang/corpusYang/Word_Seq_win3/numberTestfile"
    traiOffset = "G:/liuzhuang/corpusYang/Word_Seq_win3/numberTrainfile"
    wordVectorFile = "G:/liuzhuang/corpusYang/embedding100.vec"

    print("Get max and min position...")
    maxtest, mintest = getMaxMinOffset(testOffset)
    maxtrai, mintrai = getMaxMinOffset(traiOffset)
    max , min = max([maxtest, maxtrai]), min([mintest, mintrai])
    word_dict = generateWordVectorDict(wordVectorFile)

    offfsetVectorLengths = [25, 30, 35, 40]
    wordVectorLength = 100
    for offfsetVectorLength in offfsetVectorLengths:
        print("generate word dict and offset dict...")
        offset_dict = generateOffsetVectorDict(min, max, offfsetVectorLength)

        print("generate word + offset vector...")
        newVectorFileNametrain = generateOffsetVector(word_dict, offset_dict, traiOffset, traiContext,wordVectorLength, offfsetVectorLength, "train")
        newVectorFileNametest = generateOffsetVector(word_dict, offset_dict, testOffset, testContext, wordVectorLength, offfsetVectorLength, "test")
        mergeVector(newVectorFileNametrain, newVectorFileNametest, "Word_Seq_win3_offset_" + str(wordVectorLength + offfsetVectorLength * 2))