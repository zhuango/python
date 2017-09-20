from vector2d import *

brl = 1/2.43
print(brl)
print(format(brl, '0.4f'))

# Formatting specifier is '0.2f'. The 'rate' substring in the replacement field is
# called the field name. It’s unrelated to the formatting specifier, but determines which argument of .format() goes into that replacement field.
sss = '1 BRL = {rate:0.2f} USD'.format(rate=brl)
print(sss)

print(format(42, 'b'))
print(format(2/3, '.1%'))

from datetime import datetime
now = datetime.now()
print(format(now, '%H:%M:%S'))
print("It's now {:%I:%M %p}".format(now))

v1 = Vector2D(3, 4)
print(format(v1))

print(format(v1, '.3f'))
