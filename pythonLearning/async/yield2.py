def foo(a):
    if a:
        print("hit!")
    else:
        print("nothing!")
def generator():
    x = yield 42
    print(x)
    x = yield
    print(x)
    x = 12 + (yield 42)
    print(x)
    x = 12 + (yield)
    print(x)
    foo((yield 42))
    foo((yield))

gen = generator()

a = next(gen)
print(a) # 42

# print(100)
gen.send(100)

# print(1)
b = gen.send(1)
print(b) # 42

# print(14)
gen.send(2)

# print(15)
gen.send(3)

# foo(4)
c = gen.send(4)
print(c) # 42

