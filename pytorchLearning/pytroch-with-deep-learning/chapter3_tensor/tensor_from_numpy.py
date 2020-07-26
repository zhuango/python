import torch
import numpy

# torch tensor on cpu share the storage object with the corresponging
#  numpy object by perforam numpy() method.
points = torch.tensor([[2.2, 3.2], [1.2, 5.3], [26.76, 2.3]])
points_numpy = points.numpy()
points_numpy[0][0] = 1000.0

print(points)
print(points_numpy)


numpy_points = numpy.zeros((3, 2), dtype=numpy.float32)
points_from_numpy = torch.from_numpy(numpy_points)
numpy_points[0][0] = 1000
print(points_from_numpy)
print(numpy_points)
