# 1 The points tensor is copied to the GPU.
# 2 A new tensor is allocated on the GPU and used to store the result of the multiplication.
# 3 A handle to that GPU tensor is returned.

points = torch.tensor([[4.05, 2.2], [4.3, 7.2] [2.3, 5.7]])
# copy a tensor to GPU.
points_gpu = points.to(device='cuda:0')
# points_gpu = points.cuda(0)

# calculate on CPU.
points = 2 * points
# calculate on GPU and return a handle.
points_gpu = 2 * points.to(device='cuda')

