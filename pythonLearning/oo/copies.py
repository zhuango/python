l1 = [3, [55, 44], (7, 8, 9)]
l2 = list(l1)

print(l2 == l1)
print(l2 is l1)

print(l2[1] is l1[1])
print(l2[-1] is l1[-1])


a = [1, 2, 3, 4]
b = a[1:3]
print(b)
b[0] = 1000
print(a)
print(b)

tuplea = (1, 2,3, 4)
tupleb = tuple(tuplea)
print(tuplea is tupleb)
tuplec = tupleb[:]
print(tuplec is tupleb)
# same behavior can be observed with instances of str, bytes and frozenset.

a = frozenset((1,2, 3))
b = a.copy()
print(a is b)
