#!/ust/bin/python3

charles= {'name': 'Charles L. Dodgson', 'born': 1832}
lewis = charles
print(lewis is charles)

print(id(lewis))
print(id(charles))

lewis['balance'] = 950
print(charles)

alex = {'name': 'Charles L. Dodgson', 'born':1832, 'balance':950}
print(alex == charles)
print(alex is not charles)

# The real meaning of an object’s id is implementation-dependent. 
# In CPython, id()returns the memory address of the object, 
# but it may be something else in anotherPython interpreter. 
# The key point is that the id is guaranteed to be a unique numeric label,
# and it will never change during the life of the object
