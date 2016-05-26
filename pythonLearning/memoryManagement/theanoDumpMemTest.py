import memory_profiler
import random
import theano
from theano.misc.pkl_utils import dump, load, StripPickler

#python -m memory_profiler memory.py
#theano.misc.pkl_utils
def random_string():
    return "".join([chr(64 + random.randint(0, 25)) for _ in xrange(20)])

@profile
def create_file():
    x = [(random.random(),
          random_string(),
          random.randint(0, 2 ** 64))
         for _ in xrange(100000)]
    theano.misc.pkl_utils.dump(x, open('machin.pkl', 'w'))
    
@profile
def load_file():
    y = theano.misc.pkl_utils.load(open('machin.pkl', 'r'))
    return y
if __name__=="__main__":
    create_file()
    #load_file()