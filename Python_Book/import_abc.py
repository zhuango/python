#!/usr/bin/python
import Liu
print Liu.a, Liu.b, Liu.c

from Liu import a, b, c;
print a + ' ' + b + ' ' + ' ' + c;

c = 'Liu Zhuang';
print 'in import_abc.py: c = ' + c;
print 'in abc.py: c = ' + Liu.c;

dir(Liu)
