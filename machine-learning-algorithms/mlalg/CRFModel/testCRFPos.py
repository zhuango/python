import numpy as np
from crf import Sample, CRF

trainCorpus = "/home/laboratory/github/homeWork/machineTranslation/data/train.txt"
testCorpus  = "/home/laboratory/github/homeWork/machineTranslation/data/test.txt"

labelTableverse = {}
with open("/home/laboratory/github/homeWork/machineTranslation/data/label.txt", 'r') as f:
    index = 0
    for line in f:
        for items in line.strip().replace("  ", " ").split(" "):
            if items == "":
                continue
            word_label = items.split("/")
            if "]" in word_label[1]:
                word_label[1] = word_label[1].split("]")[0]
            if word_label[1] not in labelTableverse:
                Sample.LabelTable[index] = word_label[1]
                labelTableverse[Sample.LabelTable[index]] = index
                index += 1

with open("/home/laboratory/corpus/cn_vectorTable/cn_vectors_50.txt", 'r') as f:
    for line in f:
        items = line.strip().split(" ")
        Sample.WordsTable[items[0]] = np.array([float(elem) for elem in items[1:]], dtype=np.float)

def load(filename):
    data = []
    maxLen = 0
    with open(filename, 'r') as f:
        for line in f:
            wordSeq = []
            labels  = []
            for items in line.strip().replace("  ", " ").split(" "):
                if items == "":
                    continue
                word_label = items.split("/")
                if "]" in word_label[1]:
                    word_label[1] = word_label[1].split("]")[0]
                wordSeq.append(word_label[0])
                #print(line)
                try:
                    labels.append(labelTableverse[word_label[1]])
                except:
                    print(word_label[1])
            seqLength = len(wordSeq)
            if seqLength > maxLen:
                maxLen = seqLength
            data.append(Sample(wordSeq, labels))
    return data, maxLen

train, maxLen = load("/home/laboratory/github/homeWork/machineTranslation/data/train.txt")
test, _  = load("/home/laboratory/github/homeWork/machineTranslation/data/test.txt")

# nodeFeatureSize = len(Sample.LabelTable) + len(Sample.WordsTable)
# print(nodeFeatureSize)
crf = CRF(maxLen, 50, 50, len(labelTableverse))
crf.SGA(train[0:10], iterations=10, a0=20, validate=None)
labels = crf.Sample(train[1])
labelStrs = []
for label in labels:
    labelStrs.append(Sample.LabelTable[label])
print(labelStrs)