#!/usr/bin/python3.4
# It is best to think of a dictionary as an unordered set of key: value pairs,
# with the requirement that the keys are unique (within one dictionary)

tel = {
       'jack':4098,
       'sape':4139,
      }
tel['guido'] = 4127
print("tel = ", tel)
print("tel['jack'] = ", tel['jack'])

del tel['sape']
tel['irv'] = 4127
print("after del and add, tel = ", tel)

print(list(tel.keys()))
print(sorted(tel.keys()))
print("\'guido\' in tel: ", 'guido' in tel)
print("\'jack\' in tel: ", 'jack' not in tel)

newDict = dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
print("newDict = ", newDict)

newDict1 = {x:x**2 for x in (2, 4, 6)}
print("newDict1 = ", newDict1)

newDict2 = dict(sape=4139, guido=4127, jack=4098)
print("newDict2 = ", newDict2)

knights = {'gallahad': 'the pure', 'robin': 'the brave'}
for k, v in knights.items():
    print(k, v)

for i, v in enumerate(['tic', 'tac', 'teo']):
    print(i, v)

questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']
for q, a in zip(questions, answers):
    print("What is your {0}? It is {1}.".format(q, a))
    
for i in reversed(range(1, 10, 2)):
    print(i)

basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
for f in sorted(set(basket)):
    print(f)
    
import math
raw_data = [56.2, float('NaN'), 51.7, 55.3, 52.5, float('NaN'), 47.8]
filtered_data = []
for value in raw_data:
    if not math.isnan(value):
        filtered_data.append(value)
        
print("raw_data = ", raw_data)
print("filtered_data = ", filtered_data)

import collections

model = collections.defaultdict(lambda: 1)
print(model["erw"])

import sys
from collections import OrderedDict
newDict = OrderedDict((x, y) for x, y in zip([1, 2,3, 4], [5, 6, 7,8]))
for key in newDict:
    sys.stdout.write(str(key) + " " + str(newDict[key]) + "\n")
