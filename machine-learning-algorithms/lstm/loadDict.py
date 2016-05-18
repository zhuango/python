
# dictionary format: word postive/negative number
def loadDict(dictName):
    # dict format relative.
    wordPos = 0
    priorpolarityPos = 1
    postiveTag = "positive"
    negativeTag = "negative"
    
    wordDict = {}
    
    with open(dictName, "r") as dictFile:
        for line in dictFile:
            attributes = line.strip().split(" ")
            if(attributes[priorpolarityPos] == postiveTag):
                wordDict[attributes[wordPos]] = True;
            else:
                wordDict[attributes[wordPos]] = False;
    return wordDict

def loadDictYang(dictName, wordpos):
    # dict format relative.
    wordPos = wordpos
    
    wordDict = {}
    
    with open(dictName, "r") as dictFile:
        for line in dictFile:
            attributes = line.strip().split(" ")
            wordDict[attributes[wordPos]] = True;
    return wordDict

def loadPriorpolarityPosList(wordlist):
    # dict format relative.
    priorpolarityPos = 1
    isSentiTag = "1"
    spliter = " "
    
    wordDict = []
    linenumber = 0
    wordDict.append(True)# let the index be same as linenumber
    
    with open(wordlist, "r") as dictFile:
        for line in dictFile:
            attributes = line.strip().split(spliter)
            if(attributes[priorpolarityPos] == isSentiTag):
                wordDict.append(True);
            else:
                wordDict.append(False);
    return wordDict

def formatDict(dictfileName, wordindex, priIndex, positiveTag):
    newDictFileName = dictfileName + ".newDict"
    
    newPostiveTag = "positive"
    newNegativeTag = "negative"
    
    newDict = open(newDictFileName, "w")
    oldDict = open(dictfileName, "r")
    for line in oldDict:
        attributes = line.strip().split(" ")
        if(attributes[priIndex] == positiveTag):
            newDict.write(attributes[wordindex] + " " + newPostiveTag + "\n")
        else:
            newDict.write(attributes[wordindex] + " " + newNegativeTag + "\n")
    
if __name__ == "__main__":
    # dictfilename = "G:/liuzhuang/data/music_wordList.txt"
    # wordDicti = loadPriorpolarityPosList(dictfilename)
    # count = 0;

    # print(wordDicti[16])
    yangDict = loadDictYang("/home/laboratory/corpus/Eng_Jixing", 0)
    newDict = loadDictYang("/home/laboratory/corpus/en/CHIENDict.txt", 0)
    print("words which lost in NEW dict.These words are in old dict.")
    for key in yangDict:
        if(key not in newDict):
            print(key)
    print("###########################################")
    print("words which lost in OLD dict.These words are in new dict.")
    for key in newDict:
        if(key not in yangDict):
            print(key)
    formatDict("/home/laboratory/corpus/Eng_Jixing", 0, 2, "p")
    formatDict("/home/laboratory/corpus/CN_Jixing.txt", 1, 2, "p")