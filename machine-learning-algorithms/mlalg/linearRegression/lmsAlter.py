import numpy as np
import pylab as pl

m = 1000
randomGen = np.random

x = randomGen.random(size = m) * 100
x = sorted(x)

y = randomGen.random(size = m) * 1000
y = sorted(y)

alpha = 0.001

thata0 = 1
thata1 = 1

for i in range(0, m):
    error = y[i] - (thata0 + thata1 * x[i]);
    thata0 += alpha / m * error * 1
    thata1 += alpha / m * error * x[i]
    
    
print("thata0 = ")
print(thata0)
print(", thata1 = ")
print(thata1)

tmpx = np.linspace(0, 100., 100000)
pl.plot(x, y, '.k')
pl.plot(tmpx, thata0 + thata1 * tmpx)
pl.show()