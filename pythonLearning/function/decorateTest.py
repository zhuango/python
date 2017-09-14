#!/usr/bin/python

def decorate(func):
    def fff():
        print("dddddddddddd")
    return fff

@decorate
def target():
    print("target")

target()
