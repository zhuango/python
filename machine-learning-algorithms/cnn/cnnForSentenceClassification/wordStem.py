def generatewordStemDoc(document, wordStem_dict):
    with open(document, "r") as doc:
        newDoc = open(document + ".stem", "w")
        for line in doc:
            newLine = ""
            words = line.strip().split(" ")
            for word in words:
                if word in wordStem_dict:
                    newLine += wordStem_dict[word] + " "
                else:
                    newLine += word
            newDoc.write(newLine.strip() + "\n")
        newDoc.close()
def generateWordStemDict(wordStem):
    wordStem_dict = {}
    with open(wordStem, "r") as f:
        for line in f:
            strs =line.strip().split(' ')
            wordStem_dict[str.lower(strs[0])] = line[len(strs[0]) + 1:].strip()
    return wordStem_dict

def mergeWordStem(wordStemA, wordStemB):
    word_dict = {}
    newVectorFile = wordStemA + "_" + wordStemB
    with open(wordStemA, "r") as f:
        for line in f:
            strs =line.strip().split(' ')
            word_dict[str.lower(strs[0])] = line[len(strs[0]) + 1:].strip()
    with open(wordStemB, 'r') as f:
        for line in f:
            strs =line.strip().split(' ')
            word_dict[str.lower(strs[0])] = line[len(strs[0]) + 1:].strip()
    with open(newVectorFile, "w") as f:
        for word in word_dict:
            f.write(word + " " + word_dict[word] + "\n")
    return newVectorFile

#mergeWordStem("testWordStem.txt", "trainWordStem.txt")
wordStem_dict = generateWordStemDict("WordStem.txt")
generatewordStemDoc("train_word.cnn", wordStem_dict)
generatewordStemDoc("test_word.cnn", wordStem_dict)