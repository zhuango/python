#!/usr/bin/env python
from multiprocessing import Process
import os
import time

def sleeper(name, seconds, st):
    for i in range(20):
        print("PID: " + str(os.getpid()) + " " + str(i))
        time.sleep(seconds)
        
    print "Done sleeping"


if __name__ == '__main__':
   print "in parent process (id %s)" % os.getpid()
   for i in range(3):
        print "in parent process (id %s)" % os.getpid()
        p = Process(target=sleeper, args=('bob', 1 ,"SDF"))
        
        p.start()
        print(p.ident)
        # p.join()