#!/usr/bin/python3

def f1(a):
    print(a)
    print(b)

# f1(3)

c = 2
def f2(a):
    print(a)
    print(c)
f2(4)

def f3(a):
    print(a)
    print(c)
    c = 10
# f3(10) 

def f4(a):
    global c
    print(a)
    print(c)
    c = 10
f4(10)
f4(11)
