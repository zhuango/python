import torch

# Tensor metadata.
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

# got a offset of 2 in the storage object.
second_point = points[1]
print(second_point.storage_offset())
print(second_point.size())

# transpose is performed by just swap the stride.
print(points.stride())
points = points.transpose(0, 1)
print(points.stride())

# clone a tensor.
points_clone = points.clone()
points_clone[0][0] = 100
print(points_clone)
print(points)

# contiguous tensors.
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_t = points.t()
print(points_t)

print(points_t.storage())
print(points_t.stride())
# create another tensor storage object.
points_t_cont = points_t.contiguous()
print(points_t_cont)
print(points_t_cont.stride())
print(points_t_cont.storage())

points_t_cont[0][0] = 1000
print(points_t_cont)
print(points)
print(points_t)