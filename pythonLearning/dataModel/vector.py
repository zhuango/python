from math import hypot

class Vector:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

    # If you only implement one of these special methods, choose __repr__, because when
    # no custom __str__ is available, Python will call __repr__ as a fallback.
    # one(__repr__) used for debugging and logging, another(__str__) for presentation to end users
    def __repr__(self):
        return "Vector(%r, %r)" % (self.x, self.y)
    def __str__(self):
        return "vector({},{})".format(self.x, self.y)
    def __abs__(self):
        return hypot(self.x, self.y)

    # bool(x) calls x.__bool__() and uses
    # the result. If __bool__ is not implemented, Python tries to invoke x.__len__(), and if
    # that returns zero, bool returns False. Otherwise bool returns True.
    def __bool__(self):
        return bool(abs(self))
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

if __name__ == "__main__":
    vector0 = Vector(1, 2)
    print(vector0)
    print(repr(vector0))
    print(str(vector0))

    vector1 = Vector(2, 4)
    print(repr(vector0 + vector1))
    print(repr(vector1 * 2))

    if vector1:
        print("vector1 is True.")
    vector2 = Vector(0, 0)
    if vector2:
        pritn("vector2 is True.")
    else:
        print("vector2 is False.")
    
    print(2 or 3)