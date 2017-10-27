from vector2d import *
# In a class named Dog, if you name an instance attribute in the form __mood (two leading underscores and zero or at most one trailing underscore), 
# Python stores the name in the instance __dict__ prefixed with a leading underscore and the class name,
# so in the Dog class, __mood becomes _Dog__mood, 
# and in Beagle it’s _Beagle__mood. This language feature goes by the lovely name of name mangling.

v1 = Vector2D(3, 4)
print(v1.__dict__)
print(v1._Vector2D__x)
v1._Vector2D__x = 0
print(v1._Vector2D__x)
