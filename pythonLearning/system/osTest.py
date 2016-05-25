#!/usr/bin/python
import os
cwd = os.getcwd()
print(cwd)
os.chdir('/home/laboratory/')
print(os.getcwd())

os.chdir(cwd)

os.system('ls')

functions = dir(os)
print(functions)

print(help(os))

##############################
import shutil

#shutil.copyfile('test', 'test_copy')
#shutil.move('test', 'test_move')

#############################
import glob

fileList = glob.glob('*.py')
print(fileList)

#############################
import sys
sys.stderr.write("liuzhuang is not a error.\n")
print(sys.argv)

############################
import re
strs = re.findall(r'\bf[a-z]*', 'which foot or hand fell fastest')
print(strs)

substrs = re.sub(r'(\b[a-z]+) \1', r'\1', 'cat in the the hat.')
print(substrs)

newStr = 'tea for too'.replace('too', 'two')
print(newStr)
