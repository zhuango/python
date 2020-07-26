import torch

# tensor is a view of a storage object which is always a one-dimensional array.
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points.storage())

points_storage = points.storage()
print(points_storage[0])

# change a elements in storage object would effect all the tensor object that refer it.
points_storage[0] = 100
print(points)