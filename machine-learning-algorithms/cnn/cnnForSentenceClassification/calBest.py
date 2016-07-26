#!/usr/bin/python3
import codecs
def calBest(resultFile, label):
    best = 0.0
    with codecs.open(resultFile, 'r', 'utf-16') as results:
        for line in results:
            newline = line.strip()
            #print(line)#
            if line.startswith(label):
                current = float(line.strip().split(" ")[1])
                if current > best:
                    best = current
    return best



label = "F-score"
best = calBest("../cnnresult.txt", label)
print("best " + label + " : " + str(best))


