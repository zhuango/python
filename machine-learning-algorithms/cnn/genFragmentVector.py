import argparse
import numpy

lengthOfFragment = 5

def genVectorTable(vectorsDict):
    ####################################
    linenumber = 0
    ####################################
    vectors = {}
    with open(vectorsDict, 'r') as f:
        for line in f:
            linenumber += 1   #
            if (linenumber % 5000 == 0): print(linenumber) #
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]
    return vectors

def generate(wordslist, vectors, dimension):
    parser = argparse.ArgumentParser()
    parser.add_argument('--vo', default=wordslist, type=str)
    parser.add_argument('--di', default=dimension, type=str)
    args = parser.parse_args()
    
    wordslist = args.vo
    dimension = int(args.di)
    # with open(args.vo, 'r') as f:
    #     for line in f:
    #         for word in line.rstrip().split(' ')
    #             words.append(word)
    # words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    
    ####################################
    linenumber = 0
    notFoundCount = 0
    ####################################
    
    vecfile = open(wordslist +"_" + str(dimension) + ".vector", "w")
    with open(wordslist, 'r') as f:
        for line in f:
            fragmentVector=""
            linenumber += 1  # ################################
            if (linenumber % 5000 == 0): print(linenumber)# 
            for word in line.rstrip().split(' '):
                try:
                    for elem in vectors[word]:
                        fragmentVector = fragmentVector + str(elem) + " "
                except Exception:
                    notFoundCount += 1 ###
                    # print(word)
                    # random
                    for elem in numpy.random.rand(dimension) - 0.5:
                        fragmentVector = fragmentVector + str(elem) + " "
            vecfile.writelines(fragmentVector + "\n")
    print("There are " + str(notFoundCount) + " words witch are not found.")
    
    
from multiprocessing import Process
import os
import time

if __name__ == "__main__":

    classes = ["book", "music", "dvd"]
    wordDimensions = [50]#, 100]
    languages = ["en", "cn"]
    
    #corpusPath = "G:/liuzhuang/corpus_newDict_AddMoreNegativeWords/"
    corpusPath = "/home/laboratory/corpus/"
    #vectorTablePath = "G:/liuzhuang/corpus/"
    vectorTablePath = "/home/laboratory/corpus/"
    
    for wordDimension in wordDimensions:
        for language in languages:
            vectorsDict = vectorTablePath + "/"+language+"_vectorTable/"+language+"_vectors_"+ str(wordDimension) +".txt"
            vectors = genVectorTable(vectorsDict)
            for clas in classes:
                
                wordslist = corpusPath + language + "/label_"+clas+"_new.txt.extract"
                p = Process(target=generate, args=(wordslist, vectors, wordDimension))
                p.start()
                print(str(wordDimension) + " " + language + " " + clas + " is running. PID: " + str(p.ident))
                #p.join()
                
                wordslist = corpusPath + language + "/test_"+clas+"_new.txt.extract"
                p1 = Process(target=generate, args=(wordslist, vectors, wordDimension))
                p1.start()
                print(str(wordDimension) + " " + language + " " + clas + " is running. PID: " + str(p1.ident))
                p1.join()
