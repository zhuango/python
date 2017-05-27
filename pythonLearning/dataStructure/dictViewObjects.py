#!/usr/bin/python3
dishes = {'egg':2, 'sausage':1, 'bacon':1, 'spam':500}
keys = dishes.keys()
values = dishes.values()

n = 0
for val in values:
    n += val
print(n)

listKeys = list(keys)
print(listKeys)

del dishes['egg']
del dishes['sausage']
print(list(keys))

setAnd = keys & {'egg', 'bacon', 'salad'}
print(setAnd)

setOr = keys ^ {'sausage', 'juice'}
print(setOr)

# has no egg and sausage
print(dishes)