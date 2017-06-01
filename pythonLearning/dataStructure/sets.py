#!/usr/bin/python3

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
print("a ^ b", a ^ b) # letters in a or b but not both

a = {x for x in 'abracadabra' if x not in 'abc'}
print("a = ", a)

from unicodedata import name

specialChar = { chr(i) for i in range(32, 256) if 'SIGN' in name(chr(i), '')}
print(specialChar)