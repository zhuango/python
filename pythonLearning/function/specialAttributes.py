class C:pass
obj = C()

def func(): pass
specialAttributes = sorted(set(dir(func)) - set(dir(C)))
print(specialAttributes)
