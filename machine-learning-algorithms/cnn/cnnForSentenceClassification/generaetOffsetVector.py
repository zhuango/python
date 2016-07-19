# /usr/bin/python2.7
import numpy
import math
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
def generateOffsetVector(wordVectorFile, offsetName, contextName, vectorLength):
    word_dict={}
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
            i = 0
            for offset in line.strip().split(" "):
                wordStr = words[i] + offset
                newsent += wordStr + " "
                vectorStr = word_dict[words[i]]
                for value in numpy.random.uniform(-0.25,0.25,vectorLength * 2):
                    vectorStr += " " + str(value) 
                word_dict_withOffset[wordStr] = vectorStr
                i+=1
        
            newcontext.write(newsent.strip() + "\n")
        context.close()
    
    newVectorFile = open(wordVectorFile + ".extent", "w")
    for word in word_dict_withOffset:
        newVectorFile.write(word + " " + word_dict_withOffset[word] + "\n")
    newVectorFile.close()


if __name__ == "__main__":
    traiContext = "/home/laboratory/corpusYang/Ltrainword.context"
    testContext = "/home/laboratory/corpusYang/Ltestword.context"

    testOffset = "/home/laboratory/corpusYang/numberTestfile"
    traiOffset = "/home/laboratory/corpusYang/numberfile"
    wordVectorFile = "/home/laboratory/corpusYang/finalWordEmbeddings_50dim.txt"

    #generateOffsetVector(wordVectorFile, traiOffset, traiContext, 5)
    #generateOffsetVector(wordVectorFile, testOffset, testContext, 5)
    # max, min = getMaxMinOffset(testOffset)
    # generateOffsetVector(testOffset + ".vector", min,max ,5)

    # max, min = getMaxMinOffset(traiOffset)
    # generateOffsetVector(traiOffset + ".vector", min,max ,5)