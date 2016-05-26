import copy
import memory_profiler
import gc
import sys
#python -m memory_profiler memory.py
@profile
def function():
    x = list(range(1000000))
    y = copy.deepcopy(x)
    print(sum([sys.getsizeof(id(elem)) for elem in x]))
    
    del x
    gc.collect()
    return y

if __name__ == "__main__":
    function()