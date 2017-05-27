#!/usr/bin/python3

tt = (1, 2, (30, 40))
hashId = hash(tt)
print(hashId)

tt1 = (1, 2, [21, 12])
# Traceback (most recent call last):
#  File "canHash.py", line 8, in <module>
#    hashId = hash(tt1)
#TypeError: unhashable type: 'list'

#hashId = hash(tt1)

tf = (1,2, frozenset([30, 40]))
print(hash(tf))