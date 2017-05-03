#!/usr/bin/python3

import torch
import torch.autograd as autograd
import torch.nn.functional as F
data = autograd.Variable(torch.randn(5))
print(data)
print(F.softmax(data))
print(F.softmax(data).sum())
print(F.log_softmax(data))