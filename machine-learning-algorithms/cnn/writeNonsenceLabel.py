import numpy
import argparse

def writeNonsenceLabel(fileName, countNeedToAdd):
    fragmentLenth = 5
    dimension = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default=fileName, type=str)
    parser.add_argument('-c', default=countNeedToAdd, type=str)
    args = parser.parse_args()
    fileName = args.f
    countNeedToAdd = int(args.c)
    with open(fileName, "a") as f:
        for i in range(countNeedToAdd):
            if numpy.random.rand(dimension) - 0.5 > 0:
                labelStr = "1"
            else:
                labelStr = "0" 
            f.write(labelStr + "\n")
            
if __name__ == "__main__":
    writeNonsenceLabel("", 0)