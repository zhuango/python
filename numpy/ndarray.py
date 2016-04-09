import numpy as np 

a = np.arange(15).reshape(3, 5)
print(a)
print(a.shape)
print(a.ndim)
print(a.dtype.name)
print(a.itemsize)
print(a.size)
print(type(a))

b = np.array([6, 7, 8])
print(b)
print(type(b))

c = np.array([1.2, 3.5, 5.1])
print(c.dtype)

d = np.array([[1.5,2,3], [4,5,6]])
print(d)

e = np.array([(1.5,2,3), (4,5,6)])
print(e)

f = np.array( [ [1,2], [3,4] ], dtype=complex )
print(f)

zeros = np.zeros((3, 4))
print(zeros)

ones = np.ones((2, 3, 4), dtype = np.int16)
print(ones)

empty = np.empty((2, 3))
print(empty)

seq10_30Width_5 = np.arange(10, 30, 5)
print(seq10_30Width_5)

#less than 2
seq1 = np.arange(0, 2, 0.3)
print(seq1)