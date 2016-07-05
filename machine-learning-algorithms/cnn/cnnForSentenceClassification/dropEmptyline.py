
def DropEmptyLine(filename):
    newFilename = filename + ".dropEmptyLine";
    with open(filename, "r") as orignalFile:
        newFile = open(newFilename, "w");
        for line in orignalFile:
            if line == "\n":
                continue;
            else:
                newFile.write(line);
        newFile.close();
if __name__ == "__main__":
    DropEmptyLine("/home/laboratory/Desktop/paperAndCode/CNN_sentence-master/yang/Ltestword.context")
    DropEmptyLine("/home/laboratory/Desktop/paperAndCode/CNN_sentence-master/yang/Ltrainword.context")