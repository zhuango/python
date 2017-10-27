from vector2d_v3 import *


v1 = Vector2D(3, 4)
print(v1.__slots__)

# You must remember to redeclare __slots__ in each subclass, since the inherited attribute is ignored by the interpreter.
# • Instances will only be able to have the attributes listed in __slots__, unless you include '__dict__' in __slots__ — but doing so may negate the memory saving.
# • Instances cannot be targets of weak references unless you remember to include '__weakref__' in __slots__.

import weakref

wr = weakref.ref(v1)

