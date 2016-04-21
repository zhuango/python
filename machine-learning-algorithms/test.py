import numpy
import theano
from theano import config
from collections import OrderedDict

print(config.floatX)

import imdb

datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}
print(datasets['imdb'][1])

def firstn(n):
    num = 0
    while num < n:
        yield num
        num += 1
sumOfFirstN = sum(firstn(10000))

idx = range(10)
print(test[0][n] for n in idx)

for i in (2 * n for n in idx):
    print(i)

print(sum(n for n in idx))

array = [n for n in idx]
print(numpy.max(array))

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

def DicItems(dict):
    return [(k, dict[k]) for k in dict.keys()]
    
for key, value in DicItems(regularDict):
    print(key)
    print(value)
