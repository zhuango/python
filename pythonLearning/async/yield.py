#!/usr/bin/python3

a = 0
def yieldTest():
    global a
    print("#yield 10")
    yield 10
    print("#a = yield")
    a = yield 1000# yield nothing, when you call next send(data), data will return to a.
    print("#" + str(a))
    print("#yield a")
    yield a
    #for i in range(a):
    #    yield i

gen = yieldTest()
b = next(gen)
print("b = " + str(b))

c = gen.send(None)
print(c)

c = gen.send(0.2)
print("a = " + str(a))
print("c = " + str(c))

gen = yieldTest()
for i in gen:
    print(i)
