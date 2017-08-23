#!/usr/bin/python3
import torch
from torch.autograd import Variable
import torch.nn as nn

a = Variable(torch.randn(4, 5))
b = Variable(torch.randn(2, 3))
c = nn.Parameter(torch.randn(2, 2))

print(a)
print(b)
print(c)
torch.save([a,b,c], './a')
la,lb,lc = torch.load('./a')
print(la)
print(lb)
print(lc)
