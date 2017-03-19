import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def multivariateDensity(x, mu, sigma):
    d = mu.shape[0]

    density = 1.0 / ((2.0*np.pi) ** (d / 2.0))
    density *= 1 / (np.linalg.det(sigma) ** 0.5)
    dur = x - mu
    density *= np.exp(-0.5 * np.dot(np.dot(dur.T, np.linalg.inv(sigma)), dur))
    return density

mu = np.array([0, 0], dtype=np.float32).reshape((2, 1))
sigma = np.identity(2)

figure =plt.figure()
ax=figure.add_subplot(111,projection='3d')

X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
dataList = []
density = []
for i in range(32):
    for j in range(32):
        sample = np.array([X[i][j], Y[i][j]], dtype=np.float32).reshape((2, 1))
        density.append(multivariateDensity(sample, mu, sigma))
density = np.array(density).reshape((32, 32))

ax.plot_surface(X, Y, density)
plt.show()
#data = np.array([X0, X1]).reshape(2, 
#print(data.shape)
#mu = np.array([0.0, 0.0]).reshape(1, 2)
#sigma = np.identity(2)
#plt.plot_surface(X0, X1, , 'x')
#plt.show()