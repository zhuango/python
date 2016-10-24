#!/usr/bin/python

def bubble(nums):
    n = len(nums)
    lastStart = 0
    lastEnd = n - 1
    numPair = n - 1
    firstSwap = False
    didswitch = True
    while(didswitch):
        didswitch = False
        firstSwap = True
        i = lastStart
        while i < numPair:
            if nums[i] > nums[i + 1]:
                temp = nums[i]
                nums[i] = nums[i + 1]
                nums[i + 1] = temp
                didswitch = True
                lastEnd = i
                if i > 0 and firstSwap:
                    firstSwap = False
                    lastStart = i - 1
            i += 1
        numPair = lastEnd
array = [87, 9, 8, 6, 3, 2, 123, 43, 127, 23, 67, 10]
bubble(array)
for item in array:
    print(item)
