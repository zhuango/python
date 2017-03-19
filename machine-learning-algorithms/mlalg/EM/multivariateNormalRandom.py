import matplotlib.pyplot as plt

import numpy as np
mean = [-1, -1]
cov = [[0.1, 0], [0, 1]]
x0, y0 = np.random.multivariate_normal(mean, cov, 500).T

mean = [1, 1]
cov = [[1, 0], [0, 0.1]]
x1, y1 = np.random.multivariate_normal(mean, cov, 500).T

plt.figure(1)
plt.clf()
plt.plot(x0, y0, 'r.')
plt.plot(x1, y1, 'b.')
plt.axis('equal')
plt.show()