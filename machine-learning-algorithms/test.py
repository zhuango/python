import numpy
import theano
import theano.tensor as T
from theano import config
from collections import OrderedDict

print(config.floatX)

import imdb

# dict key(string) -> value( tuple(two functions) )
datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}
print(datasets['imdb'][1])

# generator
def firstn(n):
    num = 0
    while num < n:
        yield num
        num += 1
sumOfFirstN = sum(firstn(10000))

idx = range(10)
print(test[0][n] for n in idx)# a object of generator
print("###########################")
for i in (2 * n for n in idx):
    print(i)

print(sum(n for n in idx))

array = [n for n in idx]
print(numpy.max(array))

# Dictionary that remembers insertion order
params = OrderedDict()
params["liu"] = 123234
params[123] = 123
print(params[123])
print(params["liu"])

regularDict = {
                "liu": 234,
                21: 123
              }
print(regularDict["liu"])
print(regularDict[21])

# iterate dict.
def DicItems(dict):
    return [(k, dict[k]) for k in dict.keys()]
    
for key, value in DicItems(regularDict):
    print(key)
    print(value)

# matrix.shape is the shape of matrix i.e [3 2]
# matrix.flatten() is put all emelents in a vector, i.e [1, 2, 3, 4, 5, 0]
# matrix.reshape() 
x = T.matrix('x', dtype='int64')

shape, flatten, reshape = x.shape, x.flatten(), x.reshape([2, 3])
getShape = theano.function(inputs=[x], outputs=[shape, flatten, reshape])
print(getShape([[1, 2], [3, 4], [5, 0]]))# return [3 2], the shape of given matrix.

#
mask = T.matrix('mask', dtype = config.floatX)
something = mask[:, :, None]
getSome = theano.function(inputs=[mask], outputs=something)
print(getSome([[1, 2], [3, 4], [5, 0]]))
print(getSome([[1, 2], [3, 4], [5, 0]]))