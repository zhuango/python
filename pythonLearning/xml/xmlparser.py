import xml.etree.ElementTree as ET

def loadDict(wordVectorFile):
    wordVectors = {}
    with open(wordVectorFile, 'r') as f:
        for line in f:
            cleanLine = line.strip()
            word = cleanLine.split(" ")[0]
            wordVectors[word] = cleanLine[len(word):]
    return wordVectors
def generateInput(wordVectors, corpus, contxtWordsVector, aspectWordsVector, labels, positions, sentLengths):
    contxtWordsF  = open(contxtWordsVector, 'w')
    aspectWordsF  = open(aspectWordsVector, 'w')
    labelsF             = open(labels, 'w')
    positionsF          = open(positions, 'w')
    sentLengthsF        = open(sentLengths, 'w')

    tree = ET.parse(corpus)
    root = tree.getroot()
    for sentenceInfo in root:
        pass
        #text
        sentence = sentenceInfo[0].text.strip()
        sentenceWords = sentence.split(" ")
        sentLength = len(sentenceWords)
        positions = [i + 1 for i in range(sentLength)]
        #aspect Terms "{'polarity': 'neutral', 'to': '45', 'term': 'cord', 'from': '41'}"
        aspectTerms = []
        for aspectInfo in sentenceInfo[1]:
            aspectTerms.append(aspectInfo.attirb)
            sent = sent[:2] + sent[5 + 1:]
corpusDir           = '/home/laboratory/memoryCorpus/'
wordVectorDir       = corpusDir + 'glove_300d.txt'
corpus              = corpusDir + 'Laptop_Train_v2.xml'
contxtWordsVector   = corpusDir + 'contxtWords'
aspectWordsVector   = corpusDir + 'aspectWords'
labels              = corpusDir + 'labels'
positions           = corpusDir + 'positions'
sentLengths         = corpusDir + 'sentLengths'

wordVectors = loadDict(wordVectorDir)
generateInput(wordVectors, corpus, contxtWordsVector, aspectWordsVector, labels, positions, sentLengths)