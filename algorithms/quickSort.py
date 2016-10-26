#!python3
def quickSort(array, start, end):
    if start < end:
        pivotEle = (array[start] + array[end]) / 2
        pivotIndex = partition(array, start, end, pivotEle)
        quickSort(array, start, pivotIndex - 1)
        quickSort(array, pivotIndex, end)

def partition(array, start, end, pivotEle):
    while(start < end):
        while(array[start] <= pivotEle and end > start):
            start += 1
        swapS(array, start, end)
        while(array[end] >= pivotEle and end > start):
            end -= 1
        swapL(array, start, end)

    return start

def swapS(array, start, end):
    while(start < end):
        if array[end] < array[start]:
            temp = array[end]
            array[end] = array[start]
            array[start] = temp
            break
        end -= 1
def swapL(array, start, end):
    while(start < end):
        if array[end] < array[start]:
            temp = array[end]
            array[end] = array[start]
            array[start] = temp
            break
        start += 1

array = [9, 8, 7, 6, 5, 4, 3, 2, 1]
quickSort(array, 0, 8)
for item in array:
    print(item)