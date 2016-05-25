
    
#!/usr/bin/env python
from multiprocessing import Process
import os
import time
import numpy
import time  
import thread

def sleeper(name, seconds, sss):
    memoryMonster[1] = 1000
    for i in range(100):
        print("PID: " + str(os.getpid()) + " " + str(i))
        time.sleep(seconds)
        
    print "Done sleeping"
    thread.exit_thread()


if __name__ == '__main__':
   print "in parent process (id %s)" % os.getpid()
   memoryMonster = numpy.ones(200000000)
   for i in range(3):
        print "in parent process (id %s)" % os.getpid()
        thread.start_new_thread(sleeper, ('bob', 1 ,memoryMonster))
        time.sleep(10)
        #   p.join()