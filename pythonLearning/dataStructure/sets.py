#!/usr/bin/python3

# init set by using {}.
# more faster than set([..])
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)

print('orange' in basket)
print('crabgrass' in basket)

a = set('abracadabra')
b = set('alacazam')
print("a = ", a)
print("b = ", b)
print("a - b", a - b) # letters in a but not in b
print("a | b", a | b) # letters in either a or b
print("a & b", a & b) # letters in both a and b

print(a)
print("a & b", a.intersection(b))
print(a)

print("a ^ b", a ^ b) # letters in a or b but not both

a = {x for x in 'abracadabra' if x not in 'abc'}
print("a = ", a)

print("___init of set____")
from dis import dis
bytecode0 = dis("{1}")
print(bytecode0)

bytecode1 = dis("set([1])")
print(bytecode1)

print("___frozenset__")
immutableSet = frozenset(range(1, 10))
print(immutableSet)

print("____set comprehensions___")

# Import name function from unicodedata to obtain character names.
# Build set of characters with codes from 32 to 255 that have the word 'SIGN' in
# their names.
from unicodedata import name
set0 = {chr(i) for i in range(32, 256) if "SIGN" in name(chr(i), '')}
print(set0)
