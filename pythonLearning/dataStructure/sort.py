#!/usr/bin/python3

fruits = ['grape', 'raspberry', 'apple', 'banana']

result = sorted(fruits)
print(result)

result = sorted(fruits, reverse=True)
print(result)

result = sorted(fruits, key=len)
print(result)

result = sorted(fruits, key=len, reverse=True)
print(result)

print(fruits)

fruits.sort()
print(fruits)