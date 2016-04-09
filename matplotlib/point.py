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
y = [1, 4, 9, 16, 25]# Make an array of y values for each x value
 
pl.plot(ares_X, prices_Y, '.k')# use pylab to plot x and y
pl.show()# show the plot on the screen