#!/usr/bin/python3

from collections import Counter

counter = Counter("aabbcds")
print(counter)

counter.update("asdfasdf")
print(counter)

result = counter.most_common(2)
print(result)
