#!/usr/bin/python3
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional

input = Variable(torch.ones(1, 1, 3, 5))
# inputFeatureMapSize, outputFeatureMapSize, kernelSize
m = nn.Conv1d(1, 1, (1, 3), stride=(1, 1), padding=(0, 1), bias=False)
# batchSize, featureMapSize, elementShape
print(input)
for para in m.parameters():
    print(para)
    print(torch.sum(para))
output = m(input)
print(output)


print("=================2d conv================")
m = nn.Conv2d(1, 1, (1, 3), stride=2, padding=(0, 0), bias=False)
input = Variable(torch.ones(1, 1, 3, 5))
print(input)
for para in m.parameters():
    print(para)
    print(torch.sum(para))
output = m(input)
print(output)

    # 2d
    # f = ConvNd(_single(stride), _single(padding), _single(dilation), False,
    #            _single(0), groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled)
    # return f(input, weight, bias)

    # 1d
    # f = ConvNd(_pair(stride), _pair(padding), _pair(dilation), False,
    #            _pair(0), groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled)
    # return f(input, weight, bias)

    # 3d
    # f = ConvNd(_triple(stride), _triple(padding), _triple(dilation), False,
    #            _triple(0), groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled)
    # return f(input, weight, bias)