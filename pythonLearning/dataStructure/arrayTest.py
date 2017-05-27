from array import array
from random import random

floats = array('d', (random() for i in range(10 ** 7)))
print(floats[-1])

fp = open("floats.bin", "wb")
floats.tofile(fp)
fp.close()

floats2 = array('d')
fp = open('floats.bin', 'rb')
floats2.fromfile(fp, 10 ** 7)
fp.close()
print(floats2[-1])

a = array('d', [1, 2, 15, 4])
a = array('d', sorted(a))
print(a[-1])