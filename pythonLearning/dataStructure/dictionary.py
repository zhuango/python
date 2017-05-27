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

# the second parameter mean that the index start with 1.
for i, v in enumerate(['tic', 'tac', 'teo'], 1):
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

print("_______dict conp____")
DIAL_CODES = [
(86, 'China'),
(91, 'India'),
(1, 'United States'),
(62, 'Indonesia'),
(55, 'Brazil'),
(92, 'Pakistan'),
(880, 'Bangladesh'),
(234, 'Nigeria'),
(7, 'Russia'),
(81, 'Japan'),
]

country_code = {country: code for code, country in DIAL_CODES}
print(country_code)
print(type(country_code.items()))

print({code: country for country, code in country_code.items() if code < 66})

print("__processing missing value___")

import sys
import re
from collections import defaultdict

WORD_RE  =re.compile('\w+')
index = {}
indexDefault = defaultdict(list)
with open(sys.argv[0], encoding='utf-8') as fp:
    for line_no, line in enumerate(fp, 1):
        for match in WORD_RE.finditer(line):
            word = match.group()
            column_no = match.start() + 1
            location = (line_no, column_no)

            # occurrences = index.get(word, [])
            # occurrences.append(location)
            # index[word] = occurrences

            #index.setdefault(word, []).append(location)

            indexDefault[word].append(location)

for word in sorted(index, key=str.upper):
    print(word, index[word])

print("__imp a string key dict___")

class StrKeyDict0(dict):
    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    
    def __contains__(self, key):
        return key in self.keys() or str(key) in self.keys()

d = StrKeyDict0([('2', 'two'), ('4', 'four')])
print(d['2'])
print(d[2])
#print(d[1])

print(d.get('2'))
print(d.get(2))
print(d.get(1, 'N/A'))

print(2 in d)
print(1 in d)
