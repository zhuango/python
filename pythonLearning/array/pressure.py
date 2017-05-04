from array import array
import random
import time

n = 10000000
# arr = [i for i in range(n)]
# lis = array('l', arr)
# dic = {}
# for i in range(n):
#     dic[i] = i

# t0 = time.time()
# for i in range(10000000):
#     index = random.randint(0, n-1)
#     a = arr[index]
#     a += 1

# t1 = time.time()
# print("array: " + str(t1 - t0))


# t0 = time.time()
# for i in range(10000000):
#     index = random.randint(0, n-1)
#     a = arr[index]
#     a += 1
# t1 = time.time()
# print("list: " + str(t1 - t0))

# t0 = time.time()
# for i in range(10000000):
#     index = random.randint(0, n-1)
#     a = arr[index]
#     a += 1
# t1 = time.time()
# print("dict: " + str(t1 - t0))

a = []
b = {}

t0 = time.time()
for i in range(100000):
    #a.insert(100000, i)
    a.append(i)
t1 = time.time()
print("list append: " + str(t1 - t0))

t0 = time.time()
for i in range(100000):
    b[i] = i
t1 = time.time()
print("dict append: " + str(t1 - t0))
