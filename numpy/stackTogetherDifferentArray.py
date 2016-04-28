import numpy as np
a = np.floor(10 * np.random.random((2,2)))
print("a = ")
print(a)
b = np.floor(10 * np.random.random((2,2)))
print("b = ")
print(b)

newarray = np.vstack((a, b))
print("newarray = ")
print(newarray)

newarray2 = np.hstack((a, b))
print("newarray2 = ")
print(newarray2)

from numpy import newaxis
print("(a, b) after column_stack(a, b) = ")
print(np.column_stack((a, b)))
a = np.array([4., 2.])
b = np.array([2., 8.])
newaxisOp = a[:, newaxis]
print("newaxisOp = ")
print(newaxisOp)
print("np.column_stack((a[:,newaxis], b[:,newaxis])) = ")
print(np.column_stack((a[:,newaxis], b[:,newaxis])))
print(np.vstack((a[:,newaxis], b[:,newaxis])))

