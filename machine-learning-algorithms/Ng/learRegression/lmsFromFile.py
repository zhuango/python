import numpy
import pylab as pl

N = 97
rng = numpy.random
x = 0
y = 1

result=[]
with open('data1.txt','r') as f:
    for line in f:
        result.append(list(map(float,line.split(','))))
    print(result)

# result1=[]
# with open('a.txt','r') as f:
#     for line in f:
#         result1.append(list(map(float,line.split(','))))
#     print(result1)


thata0 = 0.#rng.random(size = 1)
thata1 = 0.#rng.random(size = 1)

alpha = 0.01

while True:
    delta0 = 0.
    delta1 = 0.
    #delta2 = 0.
    for j in range(0, N):
        error = result[j][y] - (thata0 + thata1 * result[j][x])
        delta0 += error
        delta1 += error * result[j][x]
        #delta2 += (result[j] - (thata0 + thata1 * result[j] + thata2 * result[j] * result[j])) * result[j] * result[j]
    
    if abs(delta0) < 0.01 and abs(delta1) < 0.01:
        break
    thata0 += alpha / N * delta0
    thata1 += alpha / N * delta1
    #thata2 += alpha * delta2

#draw the figure.
for i in range(0,N):
    pl.plot(result[i][x], result[i][y], '.k')# use pylab to plot x and y
    
x = numpy.linspace(0, 30., 100000)
pl.plot(x, thata0 + thata1 * x)
pl.show()# show the plot on the screen