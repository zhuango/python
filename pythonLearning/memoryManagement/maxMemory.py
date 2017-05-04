#!/usr/bin/python3

dic = []

i = 0
while True:
    dic[i] = i
    i += 1
    if i % 1000 == 0:
        print(i)
