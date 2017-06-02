#!/usr/bin/python3
import torch
from torch.autograd import Variable

# 3 dimension vector
# tensor0 length 6
# tensor1 length 1
tensor0 = Variable(torch.randn(1, 3, 6))
tensor1 = Variable(torch.randn(1, 3, 1))

K = 5
M = Variable(torch.randn(K, 3, 3))

batchTensor0 = tensor0.transpose(1, 2).expand(K, 6, 3)
batchTensor1 = tensor1.expand(K, 3, 1)

result = batchTensor0.bmm(M).bmm(batchTensor1)
print(result)
