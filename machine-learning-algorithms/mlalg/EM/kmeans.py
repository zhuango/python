import matplotlib.pyplot as plt
import numpy as np

def getLabels(data, clusters):
    label = 0
    distance = np.sum((data - clusters[0])**2)
    for i in range(len(clusters)):
        currentDistance = np.sum((data - clusters[i])**2)
        if currentDistance < distance:
            distance = currentDistance
            label = i
    return label, distance

mean = [-1, -1]
cov = [[0.1, 0], [0, 1]]
x0, y0 = np.random.multivariate_normal(mean, cov, 500).T

mean = [1, 1]
cov = [[1, 0], [0, 0.1]]
x1, y1 = np.random.multivariate_normal(mean, cov, 500).T

datas = []
for i in range(len(x0)):
    datas.append(np.array([x0[i], y0[i]], dtype=np.float32))
for i in range(len(x1)):
    datas.append(np.array([x1[i], y1[i]], dtype=np.float32))
    
datas = np.array(datas, dtype=np.float32)

clusters = np.array([[-1, 2], [2, -1]], dtype = np.float32)
labels = [0.0 for i in range(len(datas))]
k = len(clusters)
totalLoss = 0

plt.figure(1)

step = 0
while step < 50:
    currentIterationLoss = 0
    # E step
    for i in range(len(datas)):
        (label, loss) = getLabels(datas[i], clusters)
        labels[i] = label
        currentIterationLoss += loss
    if abs(currentIterationLoss - totalLoss) < 0.1:
        break
    else:
        totalLoss = currentIterationLoss
    # M step
    clusters = np.zeros_like(clusters)
    countLabeledK = [0 for i in range(k)]
    for i in range(len(datas)):
        clusters[labels[i]] += datas[i]
        countLabeledK[labels[i]] += 1
    for i in range(k):
        clusters[i] = clusters[i] / countLabeledK[i]    

    plt.clf()
    data0 = np.array([datas[i] for i in range(len(labels)) if labels[i] == 0], dtype = np.float32)
    data1 = np.array([datas[i] for i in range(len(labels)) if labels[i] == 1], dtype = np.float32)
    plt.plot(data0[:,0], data0[:,1], 'r.')
    plt.plot(data1[:,0], data1[:,1], 'b.')
    for i in range(k):
        plt.plot(clusters[i][0], clusters[i][1], 'kx')
    plt.show()

    step += 1