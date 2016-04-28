import numpy as np

def f(x, y):
    return 10 * x + y
b = np.fromfunction(f, (5, 4), dtype = int)
#call function with parameter :(0,0), (0, 1), (0, 2),...
print(b)

for row in b:
    print(row)
    
for element in b.flat:
    print(element)