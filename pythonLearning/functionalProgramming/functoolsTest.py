import functools

def add(x, y):
    return x + y

result = functools.reduce(lambda x, y: x + y, [1, 2,3 , 4, 5])
print(result)

result = functools.reduce(lambda x, y: x * y, [1, 2, 3], 100)
print(result)

result = functools.reduce(add, [1, 2, 3], 100)
print(result)

from functools import partial
def bepartial(x, y, z):
    return x + y + z
baseTwo = partial(int, base=2)
baseTwo.__doc__ = "Convert base 2 string to an int."
print(baseTwo('1001'))

myown = partial(bepartial, y = 100)
print(myown(1, z = 100))

