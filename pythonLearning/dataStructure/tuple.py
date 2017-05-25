# Tuples do double-duty: 
# they can be used as immutable lists and 
# also as records with no field names. 

t = 12345, 54321, 'hello!'
print(t)
print("t[0] = ", t[0])
print("t[1] = ", t[1])

u = t, (1, 2, 3, 4, 5)
print(u)

v = ([1, 2, 3],[3, 2, 1])
print(v)
v[0][2] = 1000
print(v)

empty = ()
singleton = 'hello',

print(len(empty))
print(len(singleton))
print(singleton)

x, y, z = t
print("x from t : ", x)
print("y from t : ", y)
print("z from t : ", z)

print("速度发送地方")