import numpy
import pylab as pl

N = 50
rng = numpy.random

ares_X = rng.random(size = N) * 100
ares_X = sorted(ares_X)

prices_Y = rng.random(size = N) * 1000
prices_Y = sorted(prices_Y)

thata0 = 0.#rng.random(size = 1)
thata1 = 0.#rng.random(size = 1)

alpha = 0.00005

while True:
    delta0 = 0.
    delta1 = 0.
    #delta2 = 0.
    for j in range(0, N):
        error = prices_Y[j] - (thata0 + thata1 * ares_X[j])
        delta0 += error
        delta1 += error * ares_X[j]
        #delta2 += (prices_Y[j] - (thata0 + thata1 * ares_X[j] + thata2 * ares_X[j] * ares_X[j])) * ares_X[j] * ares_X[j]
    
    if abs(delta0) < 0.01 and abs(delta1) < 0.01:
        break
    thata0 += alpha / N * delta0
    thata1 += alpha / N * delta1
    #thata2 += alpha * delta2

#draw the figure.
pl.plot(ares_X, prices_Y, '.k')# use pylab to plot x and y
x = numpy.linspace(0, 100., 100000)
pl.plot(x, thata0 + thata1 * x)
pl.show()# show the plot on the screen