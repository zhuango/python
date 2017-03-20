import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def getLabels(data, clusters):
    label = 0
    distance = np.sum((data - clusters[0])**2)
    for i in range(len(clusters)):
        currentDistance = np.sum((data - clusters[i])**2)
        if currentDistance < distance:
            distance = currentDistance
            label = i
    return label, distance

mean0 = [0.3, 0.2]
cov0 = [[1, 3], 
       [0.03, 1]]
x0, y0 = np.random.multivariate_normal(mean0, cov0, 500).T

mean1 = [4, 3]
cov1 = [[1, 0.03], 
       [-4, 1]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 500).T

mean2 = [9, 7]
cov2 = [[1, 3], 
       [0.03, 1]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 500).T

datas = []
for i in range(len(x0)):
    datas.append(np.array([x0[i], y0[i]], dtype=np.float32))
for i in range(len(x1)):
    datas.append(np.array([x1[i], y1[i]], dtype=np.float32))
for i in range(len(x2)):
    datas.append(np.array([x2[i], y2[i]], dtype=np.float32))
     
datas = np.array(datas, dtype=np.float32)

n = len(datas)
clusters = np.array([[-5, 0], [5, -2.5], [5, 7.5]], dtype = np.float32)
labels = [0.0 for i in range(len(datas))]
k = len(clusters)
totalLoss = 0

plt.figure(1)

step = 0
while step < 50:
    currentIterationLoss = 0
    # E step
    for i in range(n):
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
    for i in range(n):
        clusters[labels[i]] += datas[i]
        countLabeledK[labels[i]] += 1
    for i in range(k):
        clusters[i] = clusters[i] / countLabeledK[i]    

    plt.cla()
    data0 = np.array([datas[i] for i in range(len(labels)) if labels[i] == 0], dtype = np.float32)
    data1 = np.array([datas[i] for i in range(len(labels)) if labels[i] == 1], dtype = np.float32)
    data2 = np.array([datas[i] for i in range(len(labels)) if labels[i] == 2], dtype = np.float32)
    plt.plot(data0[:,0], data0[:,1], 'r.')
    plt.plot(data1[:,0], data1[:,1], 'b.')
    plt.plot(data2[:,0], data2[:,1], 'y.')
    plt.axis('equal')
    for i in range(k):
        plt.plot(clusters[i][0], clusters[i][1], 'kx')
    plt.draw()
    plt.pause(0.0001)
    step += 1

pik = np.array([0.0 for i in range(k)], dtype=np.float32)
for i in labels:
    pik[int(i)] += 1.0
for i in range(k):
    pik[i] = pik[i] / n

def multivariateDensity(x, mu, sigma):
    d = mu.shape[0]

    density = 1.0 / ((2.0*np.pi) ** (d / 2.0))
    density *= 1 / (np.linalg.det(sigma) ** 0.5)
    dur = x - mu
    density = np.exp(-0.5 * np.dot(np.dot(dur.T, np.linalg.inv(sigma)), dur))
    return density

logLikelihood = 0
gama   = np.zeros((n, k))
means  = clusters.reshape((k, 2, 1))
sigmas = np.array([np.identity(2) for i in range(k)], dtype = np.float32)

step = 0
while step < 50:
    # E step
    for i in range(n):
        wholeProb = 0
        for j in range(k):
            singleCompontProb = pik[j] * multivariateDensity(datas[i].reshape(2, 1), means[j], sigmas[j])[0][0]
            wholeProb += singleCompontProb
            gama[i][j] = singleCompontProb
        gama[i] /= wholeProb
    # M step
    for i in range(k):
        nk = 0
        new_mean  = np.zeros((2, 1))
        new_sigma = np.zeros((2, 2))
        new_pi    = 0

        for j in range(n):
            nk += gama[j][i]
            new_mean  += gama[j][i] * datas[j].reshape(2, 1)
        means[i]  = new_mean / nk

        for j in range(n):
            dur = (datas[j].reshape(2, 1) - means[i]).reshape(2, 1)
            new_sigma += gama[j][i] * np.dot(dur, dur.T)
        sigmas[i] = new_sigma / nk

        pik[i]   = nk / n
    new_likelihood = 0
    for i in range(n):
        sampleProb = 0
        for j in range(k):
            sampleProb += pik[j] * multivariateDensity(datas[i].reshape(2, 1), means[j], sigmas[j])
        new_likelihood += np.log(sampleProb)
    if np.abs(new_likelihood - logLikelihood) < 0.1:
        break
    logLikelihood = new_likelihood
    step += 1

print(means)
print(sigmas)
print(pik)

figure =plt.figure()
ax=figure.add_subplot(111,projection='3d')
X = np.arange(-10, 20, 0.25)
Y = np.arange(-10, 20, 0.25)
X, Y = np.meshgrid(X, Y)
dataList = []
density = []
for i in range(120):
    for j in range(120):
        sample = np.array([X[i][j], Y[i][j]], dtype=np.float32).reshape((2, 1))
        sampleDensity = 0
        for m in range(k):
            sampleDensity += pik[m] * multivariateDensity(sample, means[m], sigmas[m])
        density.append(sampleDensity)
density = np.array(density).reshape((120, 120))

ax.plot_surface(X, Y, density)
plt.show()