#!/usr/bin/python3

def factorial(n):
    '''return n!'''
    return 1 if n < 2 else n * factorial(n-1)
def reverse(word):
    return word[::-1]
s = reverse('testing')
print(s)

fruits = ['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana']
reverse_sort = sorted(fruits, key=reverse)
print(reverse_sort)

fact = factorial
map_list = list(map(fact, range(6)))
print(map_list)

map_list = [fact(elem) for elem in range(6)]
print(map_list)

map_filter_list = list(map(factorial, filter(lambda n:n%2, range(6))))
print(map_filter_list)

map_filter_list = [factorial(n) for n in range(6) if n % 2]
print(map_filter_list)

from functools import reduce
from operator import add
summ = reduce(add, range(100))
print(summ)

summ = sum(range(100))
print(summ)

from functools import reduce

def fact(n):
    return reduce(lambda a, b: a*b, range(1, n+1))

from operator import mul
def fact(n):
    return reduce(mul, range(1, n+1))

metro_data = [
        ('Tokyo', 'JP', 36.933, (35.689722, 139.691667)),
        ('Delhi NCR', 'IN', 21.935, (28.613889, 77.208889)),
        ('Mexico City', 'MX', 20.142, (19.433333, -99.133333)),
        ('New York-Newark', 'US', 20.104, (40.808611, -74.020386)),
        ('Sao Paulo', 'BR', 19.649, (-23.547778, -46.635833))]
from operator import itemgetter
# itemgetter(1) does the same as lambda fields: fields[1]: create a function that, given a collection, returns the item at index 1.
for city in sorted(metro_data, key=itemgetter(1)):
    print(city)

cc_name = itemgetter(1, 0)
for city in metro_data:
    print(cc_name(city))

