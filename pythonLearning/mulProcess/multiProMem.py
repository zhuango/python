#!/usr/bin/env python
from multiprocessing import Process
import os
import time
import numpy

def DoNothing(memoryMonster):
    memoryMonster[0] = 10
    for i in range(10):
        print("PID: " + str(os.getpid()) + " " + str(i))
        time.sleep(1)
    print("#######################")

def sleeper(name, seconds):
    memoryMonster = numpy.ones(200000000)
    memoryMonster[1] = 1000
    for i in range(10):
        print("PID: " + str(os.getpid()) + " " + str(i))
        time.sleep(seconds)
    p1 = Process(target =DoNothing, args=(memoryMonster, ))
    p1.start()
    p2 = Process(target =DoNothing, args=(memoryMonster, ))
    p2.start()
    print "Done sleeping"


if __name__ == '__main__':
   print "in parent process (id %s)" % os.getpid()
   for i in range(1):
        print "in parent process (id %s)" % os.getpid()
        p = Process(target=sleeper, args=('bob', 1))
        p.start()
        print(p.ident)
        time.sleep(2)
        p.join()
        print("$$$$$$$$$$$$$$$$$$$$$")