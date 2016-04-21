import theano 
import theano.tensor as T
import numpy

a = theano.shared(1)
values, updates = theano.scan(lambda:{a : a + 1}, n_steps = 10)

print(values)
print(updates)

b = a + 1
c = updates[a] + 1
f = theano.function([], [b, c], updates = updates)

f()
print(b)
print(c)
print(a.get_value())