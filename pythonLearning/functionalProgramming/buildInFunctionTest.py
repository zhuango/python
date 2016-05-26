print(all([1, 2, 3, 4]))#True
print(all([0, 2, 3, 4]))#False

print(any([1, 2,3 , 0]))#True
print(any([0, 0,0 , 0]))#False

print(ascii('我去abc'))# '\u6211\u53bbabc'
print(repr('我去abc'))# '我去abc'
print(str('我去abc'))# 我去abc

print(bin(123))#0b1111011
print(int(bin(123), base=2)) #123

def function():
    pass
print(callable(function)) # True
number = 1000
print(callable(number)) # False

print(chr(97))#a
print(chr(ord('a')))# a

print(compile('functoolsTest.py', 'functoolsTest.o', mode='exec'))
print(complex('1+2j'))

print(dir())