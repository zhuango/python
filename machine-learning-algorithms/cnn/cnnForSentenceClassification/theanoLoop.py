import theano
import theano.tensor as T
import numpy
def cal(x,t):
    return x[0, t[0][0]]
def printShared(x, pos):
    t = pos + 1
    result = cal(x, t)
    return result
values = []
index = T.imatrix()
x = T.matrix()
j = index
print(index[0][0].type)
y = theano.shared(value =numpy.arange(6).reshape((2, 3)), name= 'y')
z = theano.shared(value= y.get_value()[0] - 1, name='z')
test = printShared(x, y[::, 1:])
#z = theano.shared(value =numpy.arange(6).reshape((2, 3)), name= 'y', borrow=True)
#a = T.as_tensor_variable(numpy.ones(1))
a = numpy.ones(1)
for i in range(2):
    def func(y, b):
        global a
        a += i
        return y + i + b
    value, updates = theano.scan(func, sequences=x[T.arange(2), 0], outputs_info=a[0])
    values.append(value)
output = T.concatenate(values, 0)
f2 = theano.function(inputs=[x], outputs=test, allow_input_downcast=True)
f = theano.function([x], output)
print("#########################################")
print(y.get_value())
print(y.type)
print(z.get_value())
print("#########################################")

print(f2([[1, 2, 3], [4, 5, 6]]))
print(a)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
from theano.tensor.signal import downsample
result = downsample.max_pool_2d(input=x[::, 1:], ds=(2, 2), ignore_border=True)
maxpool = theano.function([x], result)
test = maxpool([[1, 2, 3], [4, 5, 6]])
print(test)