 import numpy as np
 
 a = np.floor(10*np.random.random((2,12)))
 print("a = ")
 print(a)
 
 print("split 'a' into 3 horizonal:")
 print(np.hsplit(a, 3))
 
 print("split 'a' into 3")