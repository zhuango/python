import theano
from theano import tensor as T
import numpy
import pylab as pl

N = 40
rng = numpy.random

ares_X = rng.random(size = N) * 100
ares_X = sorted(ares_X)
print(ares_X)

prices_Y = rng.random(size = N) * 1000
prices_Y = sorted(prices_Y)
print(prices_Y[0])

x = [1, 2, 3, 4, 5]# Make an array of x values
y = [1, 2, 3, 4, 5]
Y = [1, 0, 1, 0, 1]# Make an array of y values for each x value


for i in range(0, len(y)):
    if Y[i] == 0:
        pl.plot(x[i], y[i], 'ok')# use pylab to plot x and yield
    else:
        pl.plot(x[i], y[i], 'xk') 
pl.show()# show the plot on the screen

plotData(x, y);
# pl.plot(ares_X, prices_Y, 'ok')# use pylab to plot x and y
# pl.show()# show the plot on the screen