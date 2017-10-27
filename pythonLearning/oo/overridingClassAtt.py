from vector2d import *

a = Vector2D(3, 4)
dumpd = bytes(a)
print(dumpd)
print(len(dumpd))

a.typecode = 'f'
print(a.typecode)
dumpf = bytes(a)
print(dumpf)
print(len(dumpf))

print(Vector2D.typecode)

class ShortVector2d(Vector2D):
    typecode = 'f'
sv = ShortVector2d(1/11, 1/27)
print(sv)
print(len(bytes(sv)))
