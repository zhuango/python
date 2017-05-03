#!/usr/bin/python

import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([1., 2., 3.]), requires_grad=True)
print(x.data)

y = Variable(torch.Tensor([4., 5., 6.]), requires_grad=True)
z = x + y
print(z.data)
print(z.creator)

s= z.sum()
print(s)
print(s.creator)

s.backward()
print(x.grad)