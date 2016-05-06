import numpy
import argparse

def writeNonsenceVector(fileName, countNeedToAdd, wordDimension):
    fragmentLenth = 5
    dimension = wordDimension * fragmentLenth

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default=fileName, type=str)
    parser.add_argument('-c', default=countNeedToAdd, type=str)
    args = parser.parse_args()
    fileName = args.f
    countNeedToAdd = int(args.c)
    
    with open(fileName, "a") as f:
        for i in range(countNeedToAdd):
            vectorStr = ""
            for number in list(numpy.random.rand(dimension) - 0.5):
                vectorStr += str(number) + " "
            f.write(vectorStr + "\n")
            
            
if __name__ == "__main__":
    writeNonsenceVector("", 0)