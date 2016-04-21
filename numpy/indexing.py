import numpy as np 

x = np.arange(10)
print(x)
print(x[2])
print(x[-2])

x.shape = (2, 5)
print(x)
print(x[1, 3])
print(x[1, -1])

x.shape = (5, 2)
print(x)

# index pos:
#  0   1  2  3  4  5  6  7  8  9
# -10 -9 -8 -7 -6 -5 -4 -3 -2 -1
x = np.arange(10)
#get element with index which is in range of [a,b)
print(x[2:5]) # [2, 3, 4]
print(x[:-7]) # [0, 1, 2]
print(x[1:7:2]) # [1, 3, 5]
y = np.arange(35).reshape(5, 7)
###
# [[ 0  1  2  3  4  5  6]
#  [ 7  8  9 10 11 12 13]
#  [14 15 16 17 18 19 20]
#  [21 22 23 24 25 26 27]
#  [28 29 30 31 32 33 34]]
###
print(y)
print(y[1:5:2, ::3]) 
# row with index 1~4, step length is 2 : 1, 3
# and colum with complete index (0~6), step length is 3 : 0, 3, 6

# [[ 7 10 13]
#  [21 24 27]]

#######################################################################
# numpy arrays may be indexed with other arrays.copy of original array.
#######################################################################
x0 = np.arange(10, 1 ,-1)
print(x)# array([10,  9,  8,  7,  6,  5,  4,  3,  2])

x1 = x0[np.array([3, 3, 1, 8])]
print(x1)# array([7, 7, 9, 2])

x2 = x0[np.array([3,3,-3,8])]
print(x2)# array([7, 7, 4, 2])

x3 = x[np.array([[1,1],[2,3]])]
print(x3) # array([[9, 9],
          #        [8, 7]])