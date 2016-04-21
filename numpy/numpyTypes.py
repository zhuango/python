import numpy as np
# convert python numbers to array scalars
x = np.float32(1.0)
y = np.int_([1, 2, 4])

z = np.arange(3, dtype=np.uint8)
# convert the type integer to float by using astype function.
z.astype(float)
# convert the type integer to float by using np.typename(variable).
np.int8(z)
print(z.dtype)

d = np.dtype(int)
print(d)

print(np.issubdtype(d, int))
print(np.issubdtype(d, float))

