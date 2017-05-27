#!/usr/bin/python3
from collections import OrderedDict

ordered = OrderedDict()
ordered[1] = 'liuzhuang'
ordered[2] = 'ljj'
ordered[4] = 'sssd'

print(ordered)
# by default last = True
print(ordered.popitem(last=False))
print(ordered)
print(ordered.popitem(last=True))
print(ordered)
