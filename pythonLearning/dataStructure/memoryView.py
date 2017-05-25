#!/usr/bin/python3

# A memoryview is essentially a generalized NumPy array structure in Python itself
# (without the math). It allows you to share memory between data-structures (things like
# PIL images, SQLlite databases, NumPy arrays, etc.) without first copying. This is very
# important for large data sets.

from array import array
# h is for short signed integers.
# B is for unsigned char.

numbers =array('h', [-2, -1, 0, 1, 2])
memv = memoryview(numbers)
print(len(memv))

print(memv[0])
memv_oct = memv.cast('B')
memlist = memv_oct.tolist()
print(memlist)

memv_oct[5] = 4
print(numbers)