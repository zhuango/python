
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
    
    
if __name__ == "__main__":
    dictfilename = "G:/liuzhuang/data/music_wordList.txt"
    wordDicti = loadPriorpolarityPosList(dictfilename)
    count = 0;

    print(wordDicti[16])
