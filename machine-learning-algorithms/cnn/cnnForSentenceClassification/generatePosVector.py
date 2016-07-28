import numpy
def mergeVector(vectorFileA, vectorFileB):
    word_dict = {}
    newVectorFile = vectorFileA + "_" + vectorFileB
    with open(vectorFileA, "r") as f:
        for line in f:
            strs =line.strip().split(' ')
            word_dict[str.lower(strs[0])] = line[len(strs[0]) + 1:].strip()
    with open(vectorFileB, 'r') as f:
        for line in f:
            strs =line.strip().split(' ')
            word_dict[str.lower(strs[0])] = line[len(strs[0]) + 1:].strip()
    with open(newVectorFile, "w") as f:
        for word in word_dict:
            f.write(word + " " + word_dict[word] + "\n")
    return newVectorFile
def generatePosVector(wordVectorFile, posVectorFile, offsetName, contextName, wordVectorLength):
    word_dict={}
    pos_dict = {}
    word_dict_withOffset = {}
    with open(wordVectorFile, "r") as f:
        for line in f:
            strs =line.strip().split(' ')
            word_dict[str.lower(strs[0])] = line[len(strs[0]) + 1:].strip()
    with open(posVectorFile, "r") as f:
        for line in f:
            strs =line.strip().split(' ')
            pos_dict[str.lower(strs[0])] = line[len(strs[0]) + 1:].strip()

    with open(offsetName, "r") as f:
        context = open(contextName, "r")
        newcontext = open(contextName + ".withPos", 'w')
        for line in f:
            newsent = ""
            sent = context.readline().strip()
            words = sent.split(" ")
            i = 0
            for offset in line.strip().split(" "):
                wordStr = words[i] + "_" +offset
                newsent += wordStr + " "
                if words[i] not in word_dict:
                    wordvectorStr = ""                
                    for value in numpy.random.uniform(-0.25, 0.25, wordVectorLength):
                        wordvectorStr += str(value) + " "
                    word_dict[words[i]] = wordvectorStr.strip()
                vectorStr = word_dict[words[i]] + " " + pos_dict[offset]
                word_dict_withOffset[wordStr] = vectorStr
                i+=1
        
            newcontext.write(newsent.strip() + "\n")
        context.close()
    
    newVectorFile = open(contextName + ".vector", "w")
    for word in word_dict_withOffset:
        newVectorFile.write(word + " " + word_dict_withOffset[word] + "\n")
    newVectorFile.close()

if __name__ == "__main__":
    wordVectorFile = "/home/laboratory/Desktop/yuliao/WordEmbedding_50.txt"
    posVectorFile = "/home/laboratory/Desktop/yuliao/POSEmbedding.txt"

    testposfile = "pos/test_pos.cnn"
    traiposfile = "pos/train_pos.cnn"
    
    testContextfile = "wordseq/test_word.cnn"
    traiContextfile = "wordseq/train_word.cnn"

    #generatePosVector(wordVectorFile, posVectorFile, testposfile, testContextfile, 100)
    #generatePosVector(wordVectorFile, posVectorFile, traiposfile, traiContextfile, 100)
    #mergeVector("test_word.cnn.vector", "train_word.cnn.vector")
    mergeVector(mergeVector("a.vector", "b.vector"), "c.vector")