# prob format: "0.00 1.11"
# label format: "1"
def accuracy(probPath, labelPath):
    
    testCount = 0.0
    errorCount = 0.0
    
    negaIndex = 0
    posiIndex = 1
    
    labelFile = open(labelPath, "r")
    with open(probPath) as prob:
        for line in prob:
            testCount += 1
            label = float(labelFile.readline().strip())
            p = [float(elem) for elem in line.strip().split(" ")]
            if((p[posiIndex] > p[negaIndex] and label == 0) or
               (p[posiIndex] <= p[negaIndex] and label == 1)):
                errorCount += 1
    labelFile.close()
    
    return (1.0 - errorCount / testCount)

if __name__ == "__main__":
    clas = "dvd" # book dvd music
    prodCount = 3
    
    outputPath = "/home/laboratory/corpus/TotalOutput/100d/"+clas+"/";
    labelPath = "/home/laboratory/corpus/Serializer/test_"+clas+"_label.txt"
    bestPath = "/home/laboratory/corpus/TotalOutput/100d/"+clas+"/test_best.txt"
    print(clas)
    max = 0;
    for i in range(0, prodCount):
        acc = accuracy(outputPath + "test_prob_"+str(i)+".txt", labelPath)
        if(acc > max): max = acc
        print("iter "+ str(i) +" Accuracy: " + str(acc))
    
    acc = accuracy(bestPath, labelPath)
    print("best : " + str(acc))     
    print("Max: " + str(max))