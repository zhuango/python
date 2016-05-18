def AddZerosVectorToSent(indexFile, sentFile, numberFile, newSentFile, newindexFile, representationDim):
    index = open(indexFile, "r")
    sent = open(sentFile, "r")
    number = open(numberFile, "r")

    newSent = open(newSentFile, "w")
    newindex= open(newindexFile, "w")

    pragNumber = 0
        
    indexLine = index.readline().strip()
    indexRecord = indexLine.split(" ")
    while True:
        record = []
        pragNumber += 1
        for line in number:
            record = line.strip().split(" ")
            if record[0] == "end":
                break
        if len(record) == 0:
            break
            
        sentenceCount = int(record[1])
        for i in range(1, sentenceCount + 1):
            senVectorStr = ""
            try:
                if(int(indexRecord[0]) == pragNumber and int(indexRecord[1]) == i):
                    newindex.write(indexLine + "\n")
                    senVectorStr = sent.readline().strip()
                    newSent.write(senVectorStr.strip() + "\n")
                    
                    indexLine = index.readline().strip()
                    indexRecord = indexLine.split(" ")
                else:
                    newindex.write(str(pragNumber) + " " + str(i) + "\n")
                    for j in range(representationDim):
                        senVectorStr += "0.0 "
                    newSent.write(senVectorStr.strip() + "\n")
            except Exception:
                #print("pragNumber: " + str(pragNumber))
                newindex.write(str(pragNumber) + " " + str(i) + "\n")
                for j in range(representationDim):
                    senVectorStr += "0.0 "
                newSent.write(senVectorStr.strip() + "\n")
                
    index.close()
    sent.close()
    number.close()
    newSent.close()
    newindex.close()
    print(newSentFile + "\n" + newindexFile + " done.\n")

if __name__ == "__main__":
    representationDim = 50
    classes = ["book", "music", "dvd"]
    wordDimensions = [50, 100]
    languages = ["en", "cn"]
    
    corpusPath = "G:/liuzhuang/corpus/"
    cnnOutputPath = "G:/liuzhuang/corpus/cnn_output_test/"

    for clas in classes:
        for wordDimension in wordDimensions:
            for language in languages:
                branchPath = str(wordDimension)+"d/"+language+"/"+clas+"/"
                indexFile = cnnOutputPath + branchPath + "test_"+clas+"_new.txt.index"
                sentFile = cnnOutputPath + branchPath + "test_"+clas+"_new.txt.sent"
                numberFile = corpusPath + language + "/test_"+clas+"_new.txt.number"

                newSentFile = cnnOutputPath + branchPath + "test_"+clas+"_new.txt.sent.0"
                newindexFile = cnnOutputPath + branchPath + "test_"+clas+"_new.txt.index.0"
                
                AddZerosVectorToSent(indexFile, sentFile, numberFile, newSentFile, newindexFile, representationDim)