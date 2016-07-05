def devidePosAndNeg(dataFileName, labelFileName):
    posFileName = dataFileName + ".pos";
    negFileName = dataFileName + ".neg";
    posFile = open(posFileName, "w");
    negFile = open(negFileName, "w");

    posTag = "+1"
    negTag = "-1"

    with open(labelFileName, "r") as labelFile:
        dataFile = open(dataFileName, "r");
        for label in labelFile:
            if(label.strip() == posTag):
                posFile.write(dataFile.readline());
            else:
                negFile.write(dataFile.readline());
        dataFile.close()

if __name__ == "__main__":
    devidePosAndNeg("/home/laboratory/corpusYang/Ltrainword.context",
                    "/home/laboratory/corpusYang/Ltrainlabel.txt")
    devidePosAndNeg("/home/laboratory/corpusYang/Ltestword.context",
                    "/home/laboratory/corpusYang/Ltestlabel.txt")