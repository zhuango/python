#!/usr/bin/env python

import os

def my_fork():
    for i in range(3):
        child_pid = os.fork()
        if child_pid == 0:
            print ("Child Process: PID " + str(os.getpid()))
        else:
            print("ParentProcess: PID " + str(os.getpid()))
if __name__ == "__main__":

    my_fork()