#!/usr/bin/python3
import numpy
def generateDepTypeList(files, listFile):
    depTypes = {}
    for seqFileName in files:
        with open(seqFileName, "r") as seqFile:
            for line in seqFile:
                for deqtype in line.strip().split(" "):
                    depTypes[deqtype] += 1
    typeListFile = open(listFile, "w")
    for deqtype in depTypes:
        typeListFile.write(deptype + "\n")
    typeListFile.close()
def generateDepTypeVector(listFile, vectorFile, vectorLength):
    with open(listFile, 'r') as file:
        vectors = open(vectorFile, 'w')
        for line in file:
            vectorStr = ""
            for value in numpy.random.uniform(-1, 1, vectorLength):
                vectorStr += str(value) + " "
            vectorStr = line.strip() + " " + vectorStr.strip()
            vectors.write(vectorStr + "\n")
        vectors.close()

def generateDepTypeDict(originDictFile, newDictFile, newPosVecLength):
    newDict = {}
    with open(originDictFile, "r") as ori:
        newDict = open(newDictFile,'w')
        for line in ori:
            strs = line.strip().split(" ")
            newDictStr = ""
            for value in  numpy.random.uniform(-1, 1, newPosVecLength):
                newDictStr += str(value) + " "
            newDictStr = strs[0] + " " + newDictStr.strip()
            newDict.write(newDictStr + "\n")
        newDict.close()

def mergeVector(vectorFileA, vectorFileB):
    word_dict = {}
    newVectorFile = vectorFileA + "_" + vectorFileB
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
def generateWithDepType(wordVectorFile, depTypeVectorFile, depTypeName, contextName, wordVectorLength):
    word_dict={}
    pos_dict = {}
    word_dict_withOffset = {}
    with open(wordVectorFile, "r") as f:
        for line in f:
            strs =line.strip().split(' ')
            word_dict[str.lower(strs[0])] = line[len(strs[0]) + 1:].strip()
    with open(depTypeVectorFile, "r") as f:
        for line in f:
            strs =line.strip().split(' ')
            pos_dict[str.lower(strs[0])] = line[len(strs[0]) + 1:].strip()

    with open(depTypeName, "r") as f:
        context = open(contextName, "r")
        newcontext = open(contextName + ".withDepType", 'w')
        for line in f:
            newsent = ""
            sent = context.readline().strip()
            words = sent.split(" ")
            i = 0
            for offset in line.strip().split(" "):
                wordStr = words[i] + "_" +offset
                newsent += wordStr + " "
                if words[i] not in word_dict:
                    wordvectorStr = ""                
                    for value in numpy.random.uniform(-0.25, 0.25, wordVectorLength):
                        wordvectorStr += str(value) + " "
                    word_dict[words[i]] = wordvectorStr.strip()
                vectorStr = word_dict[words[i]] + " " + pos_dict[offset]
                word_dict_withOffset[wordStr] = vectorStr
                i+=1
        
            newcontext.write(newsent.strip() + "\n")
        context.close()
    
    newVectorFile = open(contextName + ".vector", "w")
    for word in word_dict_withOffset:
        newVectorFile.write(word + " " + word_dict_withOffset[word] + "\n")
    newVectorFile.close()

if __name__ == "__main__":
    #generateDepTypeList(["test_dep.cnn", "train_dep.cnn"], "dep.list")
    #generateDepTypeVector("dep.list", "depType_10.vector", 10)
    wordVectorFile = "G:/liuzhuang/corpusYang/embedding100.vec"
    #posVectorFile = "G:/liuzhuang/corpusYang/POSEmbedding.txt"
    depTypeVectorFile = "G:/liuzhuang/corpusYang/depType_10.vector"
    

    testDepTypefile = "dep/test_dep.cnn"
    traiDepTypefile = "dep/train_dep.cnn"
    rootPath = "G:\liuzhuang\corpusYang"
    testContextfile = "test_word.cnn"
    traiContextfile = "train_word.cnn"
    
    #generateWithDepType(wordVectorFile, depTypeVectorFile, testDepTypefile, testContextfile, 100)
    #generateWithDepType(wordVectorFile, depTypeVectorFile, traiDepTypefile, traiContextfile, 100)
    #mergeVector()

    generateDepTypeDict("G:\liuzhuang\corpusYang\depType_10.vector", "G:\liuzhuang\corpusYang\depType_100.vector", 100)