import theano
from theano.tensor import *
x = scalar('myval', dtype = 'int32')
print(x.name)

x = iscalar('myval')
print(x.name)

x = TensorType(dtype = 'int32', broadcastable = ())('myval')
print(x.name)

x = dmatrix()
print(x.name)

x = dmatrix('x')
print(x.name)

xyz = dmatrix('xyz')
print(xyz.name)

x, y, z = dmatrices(3)
print(x.name, y.name, z.name)

x, y, z = dmatrices('x', 'y', 'z')
print(x.name, y.name , z.name)

dtensor5 = TensorType('float64', (False, )*5)
x = dtensor5()
print(x.name)
z = dtensor5('z')
print(z.name)