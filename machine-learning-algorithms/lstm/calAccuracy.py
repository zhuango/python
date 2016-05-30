import os
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

    clas = "book" # book dvd music
    prodCount = 0
    
    outputPath = "G:/liuzhuang/corpus_WSR/TotalOutput/200d/book/";
    labelPath = "G:/liuzhuang/corpus_chenbro/Serializer/test_"+clas+"_label.txt"
    bestPath = outputPath+"/test_best.txt"
    if(not os.path.exists(bestPath)):
        print(clas + " test best has not been generated.")
    print(clas)
    max = 0;
    for i in range(0, prodCount):
        acc = accuracy(outputPath + "test_prob_"+str(i)+".txt", labelPath)
        if(acc > max): max = acc
        print("iter "+ str(i) +" Accuracy: " + str(acc))
    
    acc = accuracy(bestPath, labelPath)
    print("best : " + str(acc))