#!/usr/bin/python3
import bisect
import sys

HAYSTACK = [1, 4, 5, 6, 8, 12, 15, 20, 21, 21, 23, 23, 26, 29, 30]
NEEDLES = [0, 1, 2, 5, 8, 10, 22, 23, 29, 30, 31]

results = bisect.bisect(HAYSTACK, 6)
print(results)

results0 = bisect.bisect(HAYSTACK, 21)
results1 = bisect.bisect_left(HAYSTACK, 21)
print(results0, results1)

def grade(score, breakpoints=[60, 70, 80, 90], grades="FDCBA"):
    i = bisect.bisect(breakpoints, score)
    return grades[i]
grades = [grade(score) for score in [33, 99, 77, 70, 89, 90, 100]]
print(grades)