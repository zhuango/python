# Container sequences
# list, tuple and collections.deque can hold items of different types.
# Flat sequences
# str, bytes, bytearray, memoryview and array.array hold items of one type.

# Mutable sequences
# list, bytearray, array.array, collections.deque and memoryview
# Immutable sequences
# tuple, str and bytes
a = [66.25, 333, 333, 1, 1234.5]
print(a.count(333), a.count(66.25), a.count('x'))

a.insert(0, -1)
print("after insert(0, -1) = ", a)
a.append(333)
print("after append 333: ", a)

a.remove(333)
print("after remove 333 = ", a) # remove first occur

a.reverse()
print("after reverse() a = ", a)

a.sort()
print("after sort a = ", a)

print("pop: ", a.pop(), "a = ", a) # pop the tail.

squares = []
for x in range(10):
    squares.append(x**2)
print("squares = ", squares)

squares = list(map(lambda x:x **2, range(10)))
print("squares = ", squares);

squares = list(x ** 2 for x in range(10))
print("squares = ", squares)

tuples = [(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]
print("tuples = ", tuples)

vec = [[1,2,3], [4,5,6], [7,8,9]]
flat = [ele for row in vec for ele in row]
print("flat = ", flat)

matrix = [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
         ]
matrix_transpose = [[row[i] for row in matrix] for i in range(4)]
print("matrix_transpose = ", matrix_transpose)
print("list zip(*matrix)) = ", list(zip(*matrix)))

a = [-1, 1, 66.25, 333, 333, 1234.5]
print("a = ", a)
del a[0]
print("after del a[0] a = ", a)
del a[2:4]
print("after del a[2:4] a = ", a)
del a[:]
print ("after del a[:] a = ", a)

b = [-1, 1, 66.25, 333, 333, 1234.5]
print("b = ", b)
del b
# print("after del b, b = ", b) #error occur, b cannot be used.

print("_____________mul list__________")
a = [1, 2, 3]
a = a * 3
print(a)

a = [[1, 2, 3]]
a = a * 3
a[0][0] = 1000
print(a)