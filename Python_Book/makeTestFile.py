#!/usr/bin/pyhton

import os
ls = os.linesep

# get filename
fname = raw_input("input fname: ")

while True:
	if os.path.exists(fname):
		print "Error: '%s' already exist" %fname
		fname = raw_input("input fname: ")
	else:
		break

#get file content(test) lines

all = []
print "\nEnter lines ('.' by itself to quit).\n"

#loop until user terminates input
while True:
	entry = raw_input('>')
	if entry == '.':
		break;
	else:
		all.append(entry)

#write lines to file with proper line_endling
fobj = open(fname, 'w')
fobj.writelines(['%s%s' % (x, ls) for x in all])
fobj.close()

print 'DONE'
