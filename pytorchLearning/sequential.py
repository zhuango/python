#!/usr/bin/python3

import torch
import torch.nn as nn
from torch.autograd import Variable

model = nn.Sequential(
        nn.Conv2d(1, 20, 5),
        nn.ReLU(),
        nn.Conv2d(20, 64, 5),
        nn.ReLU()
        )
x = Variable(torch.FloatTensor(1, 1, 30, 2))
result = model(x)
print(result)

