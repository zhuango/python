import numpy
import theano.tensor.shared_randomstreams
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
def k_max_pool(x, k):
    """
    perform k-max pool on the input along the rows

    input: theano.tensor.tensor4
    
    k: theano.tensor.iscalar
    the k parameter

    Returns: 
    4D tensor
    """
    ind = numpy.argsort(x, axis = 3)

    sorted_ind = numpy.sort(ind[:,:,:, -k:], axis = 3)

    dim0, dim1, dim2, dim3 = sorted_ind.shape

    indices_dim0 = numpy.arange(dim0).repeat(dim1 * dim2 * dim3)
    indices_dim1 = numpy.arange(dim1).repeat(dim2 * dim3).reshape((dim1*dim2*dim3, 1)).repeat(dim0, axis=1).T.flatten()
    indices_dim2 = numpy.arange(dim2).repeat(dim3).reshape((dim2*dim3, 1)).repeat(dim0 * dim1, axis = 1).T.flatten()

    return x[indices_dim0, indices_dim1, indices_dim2, sorted_ind.flatten()].reshape(sorted_ind.shape)

a= numpy.ones((2, 1, 3, 4))
a[0][0][0][2] = 1231
a[0][0][0][1] = 123123
a[1][0][1][1] = 543
result= k_max_pool(numpy.reshape(a, (2, 1, 1, 12)), 2)
print(result)
print("##########################")
#result = numpy.reshape(result, (result.shape[0],result.shape[1], 9, 1))
b = numpy.ones((4, 5))
b[0][2] = 123
b[1][0] = 534
b[2][1] = 90
b[3][3] = 13
x = T.matrix('x')
maxpoo= downsample.max_pool_2d(input =x,ds = (3, 4), ignore_border=True)
func = theano.function([x], maxpoo, allow_input_downcast=True)
print(func(b))
# shape = (result.shape[0],result.shape[1], 9, 1)
# b = numpy.ones(shape)
# for i in range(result.shape[0]):
#     for j in range(result.shape[1]):
#         b[i,j]= result[i, j].reshape(9, 1)
        
# print(b)