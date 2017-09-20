#!/usr/bin/python3

def factorial(n):
    '''return n!'''
    return 1 if n < 2 else n * factorial(n-1)
result = factorial(42)
print(result)
print(type(result))
print(factorial.__doc__)

factorialList = list(map(factorial, range(11)))
print(factorialList)

attributes = dir(factorial)
print(attributes)
