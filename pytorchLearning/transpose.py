#!/usr/bin/python3
import torch

x = torch.randn(2, 3)
print(x.transpose(0, 1))