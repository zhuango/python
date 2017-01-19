#!/usr/bin/python3
#only in python3
from pathlib import Path

p = Path('../')
for x in p.iterdir():
    if x.is_dir():
        print(x)
    if x.is_file():
        print(str(x) + "*")
print("######################################")
def traversalDir(dirname):
    dir = Path(dirname)
    for item in dir.iterdir():
        if item.is_dir():
            traversalDir(str(item))
        if item.is_file():
            print(str(item))
traversalDir("../")
file = Path('./testf/test')
file.rename("test1")