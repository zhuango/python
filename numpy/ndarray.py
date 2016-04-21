import numpy as np
ndarray0 = np.ndarray(shape=(2, 3), dtype = float, order='F')
print(ndarray0) # random
print(ndarray0[0,1])

ndarray1 = np.ndarray(shape=(2, 3), dtype = float, order='C')
print(ndarray1) # random
print(ndarray1[0,1])