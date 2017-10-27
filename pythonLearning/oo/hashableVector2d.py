#!/usr/bin/python3
from vector2d import *
v1 = Vector2D(3, 4)

# __hash__ and __eq__ are also required
print(hash(v1))
print(set([v1]))

v2 = Vector2D(3.1, 4.2)
print(hash(v1), hash(v2))
print(set([v1, v2]))
