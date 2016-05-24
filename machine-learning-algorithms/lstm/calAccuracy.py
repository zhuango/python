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
               (p[posiIndex] < p[negaIndex] and label == 1)):
                errorCount += 1
    labelFile.close()
    
    return (1.0 - errorCount / testCount)

if __name__ == "__main__":
    outputPath = "/home/laboratory/corpus/TotalOutput/50d/dvd/";
    labelPath = "/home/laboratory/corpus/Serializer/test_dvd_label.txt"
    max = 0;
    for i in range(0, 24):
        acc = accuracy(outputPath + "test_prob_"+str(i)+".txt", labelPath)
        if(acc > max): max = acc
        print("iter "+ str(i) +" Accuracy: " + str(acc))
    print("Max: " + str(max))