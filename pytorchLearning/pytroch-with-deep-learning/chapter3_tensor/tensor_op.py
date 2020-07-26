import torch

a = torch.tensor([1.2, 3.4, 5.6], names=['channels'])
print(a['channels'])

# give a dimension name for each dimension of img_t
img_t = torch.randn(3, 5, 5)
img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
print("img_named shape ", img_named.shape)

# align a's dimension with img_t 
a = a.align_as(img_t)
print('align a ', a.shape)

# create a tensor with dtype=double and short
double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
print(short_points.dtype)

# cast dtype
double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()
# another way to perform casting
double_point = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(10 ,2).to(torch.short)

#when mixing input types, the inputs are converted to the larger type automatically.
points_64 = torch.randn(5, dtype=torch.double)
points_short = points_64.to(torch.short)
mixing_res = points_64 * points_short
print(mixing_res.dtype)

# in-place ops are thoes methods with trailing underscore.
a = torch.ones(3, 2)
a.zero_()
print(a)

