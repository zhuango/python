#!/usr/bin/python3
import torch

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

