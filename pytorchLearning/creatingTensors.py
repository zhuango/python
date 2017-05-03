#!/usr/bin/python3

import torch

V_data = [1., 2., 3.]
V = torch.Tensor(V_data)
print(V)

# Creates a matrix 
M_data = [[1., 2., 3.], [4., 5., 6]]
M = torch.Tensor(M_data)
print(M)

# Create a 3D tensor of size 2x2x2
T_data = [[[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]]]
T = torch.Tensor(T_data)
print(T)

print(V[0])
print(M[0])
print(T[0])

x = torch.randn((3, 4, 5))
print(x)
