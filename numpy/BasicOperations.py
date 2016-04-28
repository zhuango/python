import numpy as np

# element-wise
a = np.array([20, 30, 40, 50])
b = np.arange(4)
print(b)
c = a - b
print(c)
print(b ** 2)
print(10 * np.sin(a))

print(a < 35)

A = np.array([[1, 1], 
              [0, 1]])
              
B = np.array([[2, 0], 
              [3, 4]])
print(A*B)# elementwise product
          # [[2 0]
          #  [0 4]]
print(A.dot(B)) # matrix product
                #[[5 4]
                # [3 4]]

print(np.dot(A, B))

a = np.ones((2, 3), dtype = int)
b = np.random.random((2,3))
a *= 3
print(a)# [[3 3 3]
        #  [3 3 3]]

b += a
print(b) # [[ 3.01053513  3.36371987  3.48613548]
         #  [ 3.94132477  3.37503996  3.70851747]]
a += b # b is not automatically converted to integer type
print(b)

a = np.ones(3, dtype=np.int32)
b = np.linspace(0, 3.1415, 3)
print(b.dtype.name)
c = a + b
print(c)
print(c.dtype.name)

d = np.exp(c*1j)
print(d)
print(d.dtype.name)

a = np.random.random((2, 3))
print(a)
print(a.sum())
print(a.min())
print(a.max())

b = np.arange(12).reshape(3, 4)
print(b)
print(b.sum(axis =0)) # sum of each column
print(b.min(axis = 1))# min of each row
print(b.cumsum(axis = 1)) # cumulative sum along each row
print(b.cumsum(axis = 0)) # cumulative sum along each row
