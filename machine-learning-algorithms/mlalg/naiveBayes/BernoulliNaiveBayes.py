from numpy import *

def loadDataSet():
    postingList=[['my', 'my', 'dog', 'has', 'flea', \
            'problems', 'help', 'please'],
            ['maybe', 'not', 'take', 'him', \
            'to', 'dog', 'park', 'stupid'],
            ['my', 'dalmation', 'is', 'so', 'cute', \
            'I', 'love', 'him'],
            ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
            ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
            'to', 'stop', 'him'],
            ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# we treat the presence or absence of a word as a feature
# Bernoulli for each feature
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word " + word + " is not in my Vocabulary")
    return returnVec

# we treat the time of presence of a word as a feature
# Multinomial Bernoulli for each feature
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # calc p(wi|ci)
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

# Bernoulli Naive Bayes
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # sum up p(wi|ci) * p(ci). vec2Classify choose which w has contribution.
    # p(w) is the same for all ci and wi, thus we can drop it.
    # log(p(w0|ci)) + log(p(w1|ci)) + ... = log(p(w0|ci)) * p(w1|ci) * ...)
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

listPosts, listClasses = loadDataSet()
# collect all word into a set.
myVocalbList = createVocabList(listPosts)
print(myVocalbList)

myReturnVec0 = setOfWords2Vec(myVocalbList, listPosts[0])
print(myReturnVec0)

myReturnVec3 = setOfWords2Vec(myVocalbList, listPosts[3])
print(myReturnVec3)

trainMat = []
for posinDoc in listPosts:
    trainMat.append(bagOfWords2VecMN(myVocalbList, posinDoc))
print(trainMat)
p0V,p1V,pAb = trainNB0(trainMat, listClasses)
print(pAb)
print(p0V)
print(p1V)
testingNB()