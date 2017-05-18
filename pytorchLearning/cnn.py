#!/usr/bin/python3
import torch
import torch.nn as nn
from torch.autograd import Variable

# inputFeatureMapSize, outputFeatureMapSize, kernelSize
m = nn.Conv1d(1, 1, 3, stride=1, padding=1)
# batchSize, featureMapSize, elementShape
input = Variable(torch.ones(2, 1, 5))
#print(input)
for para in m.parameters():
    print(para)
output = m(input)
#print(output)

m = nn.Conv2d(1, 1, (1, 3), stride=1, padding=(0, 1))
input = Variable(torch.randn(4, 5))
print(input)
input = input.view(1, 1, 4, 5)
for para in m.parameters():
    print(para)
    output = m(input)
print(output)