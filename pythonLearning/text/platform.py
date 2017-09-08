count = open('cafe.txt', 'w', encoding='utf_8').write('café')
print(count)

chars = open('cafe.txt').read()
print(chars)

import os
size = os.stat('cafe.txt').st_size
print(size)

fp2 = open('cafe.txt', encoding='cp1252')
print(fp2)
print(fp2.encoding)
print(fp2.read())
fp2.close()


fp3 = open('cafe.txt', encoding='utf_8')
print(fp3)
print(fp3.read())
fp3.close()

fp4= open('cafe.txt', 'rb')
print(fp4)
print(fp4.read())

