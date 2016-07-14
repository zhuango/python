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


if __name__ == "__main__":
    testOffset = "/home/laboratory/corpusYang/numberTestfile"
    traiOffset = "/home/laboratory/corpusYang/numberfile"
    max, min = getMaxMinOffset(testOffset)
    generateOffsetVector(testOffset + ".vector", min,max ,10)

    max, min = getMaxMinOffset(traiOffset)
    generateOffsetVector(traiOffset + ".vector", min,max ,10)