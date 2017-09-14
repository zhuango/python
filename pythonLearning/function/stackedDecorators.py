#!/usr/bin/python3

# When two decorators @d1 and @d2 are applied to a function f in that order,
# the result is the same as f = d1(d2(f)).

def d1(func):
    print('d1')
    return func
def d2(func):
    print('d2')
    return func

@d1
@d2
def f():
    print('f')
f()
