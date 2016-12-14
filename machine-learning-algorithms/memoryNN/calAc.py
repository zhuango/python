#!/usr/bin/python
import numpy as np
printAll = True
maxAcc = 0.0
idx = 0
count = 0
for i in range(100000):
    predictionF = "/home/laboratory/memoryCorpus/result/predict_"+ str(i)+".txt"
    goldF = "/home/laboratory/memoryCorpus/test/labels"
    try:
        predictions = np.loadtxt(predictionF, np.float)
        golds       = np.loadtxt(goldF, dtype=np.float)
    except:
        break

    count = 0.0
    index = 0
    correct = 0.0
    for one_hot in predictions:
        #print(np.argmax(one_hot))
        if np.argmax(one_hot) == np.argmax(golds[index]):
            correct += 1.0
        index += 1
        count += 1
    curAcc = float(correct / count)
    if printAll:
        print("Accuracy "+ str(i) + ": " + str(curAcc))
    if curAcc > maxAcc:
        maxAcc = curAcc
        idx = i
print("Accuracy "+ str(idx) + ": " + str(maxAcc))

print(count)