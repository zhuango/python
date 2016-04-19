# ss 
import numpy

array = [[1, 2, 3, 4, 5],
         [6, 7, 8, 9, 10]]
array = array[:2]
idx = numpy.arange(len(array[0]))
numpy.random.shuffle(idx)
array = ([array[0][n] for n in idx])
print(array)
