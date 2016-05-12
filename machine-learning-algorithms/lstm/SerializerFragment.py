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

def generate(wordslist, vectors, dimension, dictPath, serializationPath):
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
    serialicationFile = open(serializationPath, "w")
    
    ####################################
    linenumber = 0
    notFoundCount = 0
    ####################################
    with open(wordslist, 'r') as f:
        for line in f:
            serialicationNumbersStr = ""
            linenumber += 1  # ################################
            if (linenumber % 1000 == 0): print(serializationPath + " generate serialization: "+str(linenumber))# 
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
    print(str(os.getpid()) + " Done, there are " + str(notFoundCount) + " words witch are not found.")
    dictFile.close()
    serialicationFile.close()
    
from multiprocessing import Process
import os
import time

def SingleProcess(wordDimension):
    corpusPath = "G:/liuzhuang/corpus/"
    classes = ["book", "music", "dvd"]
    languages = ["en", "cn"]
    vectorDicts = {}
    vectorsDict_en = corpusPath + languages[0]+"_vectorTable/"+languages[0]+"_vectors_"+ str(wordDimension) +".txt"
    vectorsDict_cn = corpusPath + languages[1]+"_vectorTable/"+languages[1]+"_vectors_"+ str(wordDimension) +".txt"
    vectorDicts["en"] = generateVectorDict(vectorsDict_en)
    vectorDicts["cn"] = generateVectorDict(vectorsDict_cn)
    
    for clas in classes:
        for language in languages:
            wordslist = corpusPath + language + "/test_"+clas+"_new.txt.extract"
            #wordslist_test = corpusPath + languages[0] + "/test_"+clas+"_new.txt.extract"

            dictPath = corpusPath +language + "/test_"+clas+"_new.txt.extract_"+str(wordDimension)+".lstmDict"
            serializationPath = corpusPath + language + "/test_"+clas+"_new.txt.extract_"+str(wordDimension)+".serialization"
            
            if(os.path.exists(dictPath)):
                continue
            
            ############
            #generate(wordslist, vectorDicts[language], wordDimension,dictPath,serializationPath)
            ############
            
            p = Process(target=generate, args=(wordslist, vectorDicts[language], wordDimension,dictPath,serializationPath))
            p.start()
            print(str(wordDimension) + " " + clas + " " + language + " is running. PID: " + str(p.ident))
            #p.join()
if __name__ == "__main__":

    wordDimensions = [50, 100]
    
    #cnnOutputPath = "G:/liuzhuang/corpus/lstm_output/"

    for wordDimension in wordDimensions:
            #SingleProcess(wordDimension,)#
            p = Process(target=SingleProcess, args=(wordDimension,))
            p.start()
            print(str(wordDimension) + "D " + " is running. PID: " + str(p.ident))
            p.join()
