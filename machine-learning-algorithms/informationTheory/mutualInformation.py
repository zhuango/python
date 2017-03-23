# /usr/bin/python3
import math
# mutual_Information(x, y) = entropy(x) - entropy(x|y) = entropy(y) - entropy(y|x)
def entropyDiscrete(variableCount, probs):
    entropy = 0.0
    for i in xrange(0, variableCount):
        entropy -= probs[i] * math.log(probs[i], 2.0)
    return entropy

def mutualInformation(wordProbs, wordLabelJointlyProbs, wordProbConditionOnNegative, wordProbConditionOnPositive):
    wordLabelMutualInformation = {}
    for word in wordProbs:
        prob = wordProb[word]

        wordEntropy = 0.0
        if prob != 0.0 and prob != 1.0:
            wordEntropy = -(prob * math.log(prob, 2.0) + (1 - prob) * math.log(1.0 - prob, 2.0))
        
        conditionEntropy = 0.0
        if wordProbConditionOnNegative[word] != 0.0:
            conditionEntropy -= wordLabelJointlyProbs[word].hasWordNegative * math.log(wordProbConditionOnNegative[word])
            conditionEntropy -= wordLabelJointlyProbs[word].noWordNegative  * math.log(1 - wordProbConditionOnNegative[word])
        if wordProbConditionOnPositive[word] != 0.0:
            conditionEntropy -= wordLabelJointlyProbs[word].hasWordPositive * math.log(wordProbConditionOnPositive[word])
            conditionEntropy -= wordLabelJointlyProbs[word].noWordPositive  * math.log(1 - wordProbConditionOnPositive[word])
        
        wordLabelMutualInformation[word] = wordEntropy - conditionEntropy
    return wordLabelMutualInformation

class Graph:
    def __init__(self):
        self.label = 0
        self.graphNumber = 0
        self.sentences = []
class Document:
    def __init__(self, documentPath):
        self.documentStream = open(documentPath, 'r')
        self.hasGraph = True
        self.currentLine = ""
        while True:
            line = self.documentStream.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith('<') and line.endswith('>'):
                self.currentLine = line
                break

    def Next(self):
        if not self.hasGraph:
            return None
        graph = Graph()
        print(self.currentLine)
        if   'p' in str.lower(self.currentLine):
            graph.label = 1
        elif 'n' in str.lower(self.currentLine):
            graph.label = 0
        while True:
            line = self.documentStream.readline()
            if not line:
                self.hasGraph = False
                break
            line = line.strip()
            if line.startswith('<') and line.endswith('>'):
                self.currentLine = line
                break
            graph.sentences.append(line)
        return graph
from collections import defaultdict
wordCountDict = defaultdict(lambda:0)
wordInPositiveSampleCountDict = defaultdict(lambda:0)
wordInNegativeSampleCountDict = defaultdict(lambda:0)

documentList = ["/home/laboratory/corpus/denoise/cn/label_book_new.txt",
                "/home/laboratory/corpus/denoise/cn/label_dvd_new.txt",
                "/home/laboratory/corpus/denoise/cn/label_music_new.txt",]
for documentFile in documentList:
    document = Document(documentFile)
    while True:
        graph = document.Next()
        if not graph:
            break
        for sentence in graph.sentences:
            for word in sentence.strip().split(" "):
                word = word.strip()
                if word == "":
                    continue
                wordCountDict[word] += 1
                if   graph.label == 0:
                    wordInNegativeSampleCountDict[word] += 1
                elif graph.label == 1:
                    wordInPositiveSampleCountDict[word] += 1

wordProb = defaultdict(lambda:0.0)
wordProbConditionOnPositive = defaultdict(lambda:0.0)
wordProbConditionOnNegative = defaultdict(lambda:0.0)

totalWordCount = float(sum(wordCountDict.values()))
for key in wordCountDict:
    wordProb[key] = wordCountDict[key] / totalWordCount

totalWordCountInNegativeSample = float(sum(wordInNegativeSampleCountDict.values()))
for key in wordInNegativeSampleCountDict:
    wordProbConditionOnNegative[key] = wordInNegativeSampleCountDict[key] / totalWordCountInNegativeSample

totalWordCountInPositiveSample = float(sum(wordInPositiveSampleCountDict.values()))
for key in wordInPositiveSampleCountDict:
    wordProbConditionOnPositive[key] = wordInPositiveSampleCountDict[key] / totalWordCountInPositiveSample

pNegative = totalWordCountInNegativeSample / totalWordCount
pPositive = totalWordCountInPositiveSample / totalWordCount
class WordLabelJointlyProb:
    def __init__(self, hasWordPositive = 0.0, hasWordNegative = 0.0, noWordPositive = 0.0, noWordNegatice = 0.0):
        self.hasWordPositive = 0.0
        self.hasWordNegative = 0.0
        self.noWordPositive = 0.0
        self.noWordNegative = 0.0
wordLabelJointlyProbs = defaultdict(lambda:WordLabelJointlyProb())

for key in wordCountDict:
    wordLabelJointlyProbs[key].hasWordNegative = pNegative * wordProbConditionOnNegative[key]
    wordLabelJointlyProbs[key].hasWordPositive = pPositive * wordProbConditionOnPositive[key]
    wordLabelJointlyProbs[key].noWordNegatice  = pNegative * (1.0 - wordProbConditionOnNegative[key])
    wordLabelJointlyProbs[key].noWordPositive  = pPositive * (1.0 - wordProbConditionOnPositive[key])

muInformation = mutualInformation(wordProb, wordLabelJointlyProbs, wordProbConditionOnNegative, wordProbConditionOnPositive)
muInformationSorted = sorted(muInformation.items(), key=lambda d:d[1], reverse=True)

muInformationFile = "muInformation.txt"
with open(muInformationFile, 'w') as muInformationStream:
    for item in muInformationSorted:
        muInformationStream.write(item[0] + " " + str(item[1]) + "\n")