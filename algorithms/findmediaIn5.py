#!/usr/bin/python
def findmediaIn5(array):
    if not len(array) == 5:
        return -1
    if array[0] > array[1]:
        array[0], array[1] = array[1], array[0]
    if array[2] > array[3]:
        array[2], array[3] = array[3], array[2]
    if array[1] > array[3]:
        array[1], array[3] = array[3], array[1]
    if array[2] > array[4]:
        array[2], array[4] = array[4], array[2]
    if array[1] > array[4]:
        array[1], array[4] = array[4], array[1]
    if array[1] > array[2]:
        return array[1]
    else:
        return array[2]

result = findmediaIn5([2, 34, 1, 5, 56])
print(result)