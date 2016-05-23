import numpy
import argparse

def genSentenceVector(numberFile, fragmentVectorFile, indexFile, sentenceVectorFile, dimension):

    psnumber = open(numberFile,"r")
    fragmentVectors = open(fragmentVectorFile, "r")
    index = open(indexFile, "w")
    sentenceVectors = open(sentenceVectorFile, "w")
    
    graphNumber = 0
    
    #################################
    fragNumber = 0
    #################################
    
    while True:
        fragCountInEachSentence = {}
        graphNumber +=1
        record = []
        for line in psnumber:
            record = line.strip().split(" ")
            if(len(record) == 0): break
            if(str.lower(record[0]) != "end"):
                lineN = int(record[1])
                if lineN in fragCountInEachSentence.keys():
                    fragCountInEachSentence[lineN] += 1
                else:
                    fragCountInEachSentence[lineN] = 1
            else:
                break
        if(len(record) == 0): break
        for sentenceN in sorted(fragCountInEachSentence.keys()):
            sentenceVector = numpy.zeros(dimension)
            for i in range(fragCountInEachSentence[sentenceN]):
                tmpfragVec = fragmentVectors.readline().rstrip().split(" ")
                fragNumber += 1
                sentenceVector += numpy.array([float(elem) for elem in tmpfragVec])
                
            sentenceVectorStr = ""
            for elem in sentenceVector:
                sentenceVectorStr += str(elem) + " "
            sentenceVectors.write(sentenceVectorStr + "\n")
            index.write(str(graphNumber) + " " + str(sentenceN) + "\n")
    print(str(fragNumber) + "\n" + indexFile + "\n" + sentenceVectorFile + " done.\n")        

import os
from multiprocessing import Process
if __name__ == "__main__":
#def genSen():
    mDimension = 50
    classes = ["book", "music", "dvd"]
    wordDimensions = [50, 100]
    languages = ["en", "cn"]
    
    corpusPath = "G:/liuzhuang/corpus/"
    cnnOutputPath = "G:/liuzhuang/corpus/cnn_output/"
    for clas in classes:
        for wordDimension in wordDimensions:
            for language in languages:
                numberFile = corpusPath+language+"/test_"+clas+"_new.txt.number"
                fragmentVectorFile = cnnOutputPath+str(wordDimension)+"d/"+language+"/"+clas+"/"+clas+"_output_50.txt"
                indexFile = cnnOutputPath+str(wordDimension)+"d/"+language+"/"+clas+"/" + "test_"+clas+"_new.txt.index"
                sentenceVectorFile = cnnOutputPath+str(wordDimension)+"d/"+language+"/"+clas+"/" + "test_"+clas+"_new.txt.sent"
                
                genSentenceVector(numberFile, fragmentVectorFile, indexFile, sentenceVectorFile, mDimension)