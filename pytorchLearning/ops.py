#!/usr/bin/python3
import torch
import torch.nn

# ops
x = torch.Tensor([1., 2., 3.])
y = torch.Tensor([4., 5., 6.])
z = x + y
print(z)

# concatenation
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
# 5x5
z_1 = torch.cat([x_1, y_1])
print(z_1)

x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
# 2x8
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))
# If one of the dimensions is -1, its size can be inferred
print(x.view(2, -1))

from torch.autograd import Variable
x = Variable(torch.randn(3, 4))
print(x)
print(torch.sum(x, 0))
print(torch.max(x))
print(torch.cat([x, x], 1))

y = torch.FloatStorage(1).float()
y = torch.randn(1, 4)
x = torch.randn(3, 4)
print(x.size())
print(y.expand_as(x))
print(x * y.expand_as(x))

softmax = torch.nn.Softmax()
result = softmax(Variable(y))
print(result)

import numpy as np
ones = np.ones((4, 1), np.int64)
ones = int(ones.argmax())
print(ones)

y = Variable(torch.LongTensor([ones]))
x = Variable(torch.randn(4, 1))
#x = x.view(1, 4)

print((np.argmax(x.data)))
loss_function = torch.nn.Softmax()
print("softmax")
print(loss_function(x))

torch.randn
z = Variable(torch.randn(5, 2))
y = Variable(torch.randn(1, 2))
print(y)
print(y.expand_as(z))

t = Variable(torch.Tensor(np.random.uniform(0.1, -0.1, (1, 5))))
print(t)

val, index = torch.max(t, 1)
val, index = t.max(0)
print(np.argmax(t.data.numpy()))