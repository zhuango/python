#!/usr/bin/python3
#only in python3
from pathlib import Path


def traversalDir(dirname):
    dir = Path(dirname)
    for item in dir.iterdir() :
        if item.is_file():
            item.rename(str(item) + '.pdf')
traversalDir("/home/laboratory/Documents/ebook/papers/ICML/2016/")
