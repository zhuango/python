#!/usr/bin/python3
import sys

def function(x, y):
    return x + y
    
seqa = [1, 2, 3, 4]
seqb = [1, 2, 3, 4, 5]
for res in map(function, seqa, seqb):
    sys.stdout.write(str(res) + " ")
print()


