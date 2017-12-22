import numpy
import matplotlib.pyplot as plt

# f1 = 2X - Y + 1
# f2 = -3X + 6 - Y
# f3 = -2X + Y + 3

def function1(X, Y):
    return 2 * X - 2 * Y + 101
def function2(X, Y):
    return -3 * X - Y + 106
def function3(X, Y):
    return -4 * X + Y + 104
 
# f1 = f2
def seperator1(X):
    return 5 * X - 5
# f1 = f3
def seperator2(X):
    return 2 * X - 1
# f2 = f3
def seperator3(X):
    return (1 / 2) * X + 1

plt.figure(1)
plt.clf()


X = numpy.random.uniform(-17, 17, 1000)
Y = numpy.random.uniform(-17, 17, 1000)

f1 = function1(X, Y)
f2 = function2(X, Y)
f3 = function3(X, Y)
y = Y * numpy.array(f1 >= f2, dtype=float)
y = y * numpy.array(f1 >= f3, dtype=float)
plt.plot(X, y, "r.")


y = Y * numpy.array(f2 > f1, dtype=float)
y = y * numpy.array(f2 >= f3, dtype=float)
plt.plot(X, y, "b.")

y = Y * numpy.array(f3 > f1, dtype=float)
y = y * numpy.array(f3 > f2, dtype=float)
plt.plot(X, y, "g.")

y = seperator1(X)
Y = y * numpy.array(function1(X, y) > function3(X, y), dtype=float)
plt.plot(X, Y, 'r.')

y = seperator2(X)
Y = y * numpy.array(function3(X, y) > function2(X, y), dtype=float)
plt.plot(X, Y, 'b.')

y = seperator3(X)
Y = y * numpy.array(function2(X, y) > function1(X, y), dtype=float)
plt.plot(X, Y, 'g.')


plt.axis([-17, 17, -17, 17])
plt.show()
