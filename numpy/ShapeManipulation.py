import numpy as np
a = np.floor(10 *np.random.random((3,4)))
print("a = ")
print(a)
print(a.shape)
print("a after raval(), flat array")
print(a.ravel())
a.shape = (6, 2)
print("a = ")
print(a)
print("a after transition")
print(a.T)

print("a return by reshape(2, 6) = ")
print(a.reshape(2, 6))
print("a after reshape(2, 6), not changed")
print(a)

a.resize(2, 6)
print("a after resize(2, 6)")
print(a)

print(a.reshape(3,-1))