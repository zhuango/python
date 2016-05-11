import argparse
import numpy

lengthOfFragment = 5

def generateVectorDict(vectorsDict):
    ####################################
    linenumber = 0
    ####################################

    vectors = {}
    with open(vectorsDict, 'r') as f:
        for line in f:
            linenumber += 1   #
            if (linenumber % 1000 == 0): print("generate vector dict: "+str(linenumber)) #
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]
    return vectors

def generate(wordslists_en, wordslists_cn, vectors_en, vectors_cn, dimension, dictPath):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--vo', default=wordslist, type=str)
    # parser.add_argument('--ve', default=vectorsDict, type=str)
    # parser.add_argument('--di', default=dimension, type=str)
    # args = parser.parse_args()
    
    # wordslist = args.vo
    # vectorsDict = args.ve
    # dimension = int(args.di)
    # with open(args.vo, 'r') as f:
    #     for line in f:
    #         for word in line.rstrip().split(' ')
    #             words.append(word)
    # words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    
    dictTable = {}
    dictIndex = 0
    dictFile = open(dictPath, "w")
    
    ####################################
    linenumber = 0
    notFoundCount = 0
    ####################################
    vectors = vectors_en
    for wordslist in wordslists_en:
        serialicationFile = open(wordslist +"_en_" + str(dimension) + ".serialization", "w")
        with open(wordslist, 'r') as f:
            for line in f:
                serialicationNumbersStr = ""
                linenumber += 1  # ################################
                if (linenumber % 1000 == 0): print("generate serialization: "+str(linenumber))# 
                for word in line.rstrip().split(' '):
                    wordVector=""
                    if(not word in dictTable.keys()):
                        if(word in vectors.keys()):
                            for elem in vectors[word]:
                                wordVector = wordVector + str(elem) + " "
                        else:
                            notFoundCount += 1 ###
                            # print(word)
                            # random
                            for elem in numpy.random.rand(dimension) - 0.5:
                                wordVector = wordVector + str(elem) + " "
                        dictIndex += 1
                        dictTable[word] = dictIndex
                        dictFile.writelines(wordVector.strip() + "\n")
                    serialicationNumbersStr += str(dictTable[word]) + " "
                serialicationFile.write(serialicationNumbersStr.strip() + "\n")
    print("There are " + str(notFoundCount) + " words witch are not found.")
    ####################################
    linenumber = 0
    notFoundCount = 0
    ####################################
    vectors = vectors_cn
    for wordslist in wordslists_cn:
        serialicationFile = open(wordslist +"_cn_" + str(dimension) + ".serialization", "w")
        with open(wordslist, 'r') as f:
            for line in f:
                serialicationNumbersStr = ""
                linenumber += 1  # ################################
                if (linenumber % 1000 == 0): print("generate serialization: "+str(linenumber))# 
                for word in line.rstrip().split(' '):
                    wordVector=""
                    if(not word in dictTable.keys()):
                        if(word in vectors.keys()):
                            for elem in vectors[word]:
                                wordVector = wordVector + str(elem) + " "
                        else:
                            notFoundCount += 1 ###
                            # print(word)
                            # random
                            for elem in numpy.random.rand(dimension) - 0.5:
                                wordVector = wordVector + str(elem) + " "
                        dictIndex += 1
                        dictTable[word] = dictIndex
                        dictFile.writelines(wordVector.strip() + "\n")
                    serialicationNumbersStr += str(dictTable[word]) + " "
                serialicationFile.write(serialicationNumbersStr.strip() + "\n")
    print("There are " + str(notFoundCount) + " words witch are not found.")
    
from multiprocessing import Process
import os
import time

def SingleProcess(wordDimension, i):
    corpusPath = "G:\\liuzhuang\\corpus\\"
    classes = ["book", "music", "dvd"]
    languages = ["en", "cn"]
    
    vectorsDict_en = corpusPath + languages[0]+"_vectorTable\\"+languages[0]+"_vectors_"+ str(wordDimension) +".txt"
    vectorsDict_cn = corpusPath + languages[1]+"_vectorTable\\"+languages[1]+"_vectors_"+ str(wordDimension) +".txt"
    vectors_en = generateVectorDict(vectorsDict_en)
    vectors_cn = generateVectorDict(vectorsDict_cn)
    
    for clas in classes:
        wordslists_en = []
        wordslists_cn = []
        wordslist_train = corpusPath + languages[0] + "\\label_"+clas+"_new.txt.extract"
        wordslist_test = corpusPath + languages[0] + "\\test_"+clas+"_new.txt.extract"
        wordslists_en.append(wordslist_train)
        wordslists_en.append(wordslist_test) # append must be in this order.
            
        wordslist_train = corpusPath + languages[1] + "\\label_"+clas+"_new.txt.extract"
        wordslist_test = corpusPath + languages[1] + "\\test_"+clas+"_new.txt.extract"
        wordslists_cn.append(wordslist_train)
        wordslists_cn.append(wordslist_test) # append must be in this order.
            
        vectorsDict = corpusPath + "fragment_" + clas +"_"+str(wordDimension) + ".lstmDict"
        if os.path.exists(vectorsDict):
            continue
        #generate(wordslists_en, wordslists_cn, vectors_en, vectors_cn, wordDimension,vectorsDict)#
        p = Process(target=generate, args=(wordslists_en, wordslists_cn, vectors_en, vectors_cn, wordDimension,vectorsDict))
        p.start()
        print(str(wordDimension) + " " + clas + " is running. PID: " + str(p.ident))
if __name__ == "__main__":

    wordDimensions = [50, 100]
    
    #cnnOutputPath = "G:\\liuzhuang\\corpus\\lstm_output\\"

    for wordDimension in wordDimensions:
            #SingleProcess(wordDimension)#
            p = Process(target=SingleProcess, args=(wordDimension, 0))
            p.start()
            print(str(wordDimension) + "D " + " is running. PID: " + str(p.ident))