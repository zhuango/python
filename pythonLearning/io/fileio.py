'''
The first argument is a string containing the filename. The second argument is another string containing a few characters describing the way in which the file will be used. mode can be 'r' when the file will only be read, 'w' for only writing (an existing file with the same name will be erased), and 'a' opens the file for appending; any data written to the file is automatically added to the end. 'r+' opens the file for both reading and writing. The mode argument is optional; 'r' will be assumed if it’s omitted.
'''

f = open('test', 'r')
print(f.read(10))
print(f.readline(), end = '')
f.close()

f2 = open('test', 'r')
for line in f2:
    print(line, end='')
print()
f2.close()

f3 = open('newfile', 'w')
numberOfChar = f3.write('This is the first line.\n')
print("write {0} bytes".format(numberOfChar))

value = ('the answer', 42)
s = str(value)
numberOfChar = f3.write(s)
print("write {0} bytes".format(numberOfChar))
f3.close()

f4 = open('newfile', 'rb+')
f4.write(b'0123456789abcdef')
f4.seek(5)
print(f4.read(1))
f4.seek(-3, 2)
print(f4.read(1))
f4.close()

with open('test', 'r') as f:
    read_data = f.read()
print(f.closed)
