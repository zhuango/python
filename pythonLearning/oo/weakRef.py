import weakref
s1 = {1, 2, 3}
s2 = s1

# This function must not be a bound method the object about to be destroyed or
# otherwise hold a reference to it.
def bye():
    print('Gone with the wind...')

# Weak references to an object do not increase its reference count.
# The object that is the target of a reference is called the referent. 
# Therefore, we say that a weak reference does not prevent the referent from being garbage collected.

ender = weakref.finalize(s1, bye)
print(ender.alive)

del s1
print(ender.alive)

del s2
print(ender.alive)

a_set = {0, 1}
wref = weakref.ref(a_set)
print(wref)
print(wref())

a_set = {2,3, 4}
print(wref() is None)
