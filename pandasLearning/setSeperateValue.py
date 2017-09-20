#!/usr/bin/python3

def setSeperateValue(values):
    for i in range(len(values)):
        v = values[i]
        floor = v // 0.5
        if v % 0.5 > 0.25:
            values[i] = (floor + 1) * 0.5
        else:
            values[i] = floor * 0.5