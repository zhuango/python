from operator import methodcaller
s = 'The time has come'
upcase = methodcaller('upper')
result = upcase(s)
print(result)

hiphenate = methodcaller('replace', ' ', '-')
result = hiphenate(s)
print(result)

