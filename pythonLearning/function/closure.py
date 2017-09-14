#!/usr/bin/python3

class Averager():
    def __init__(self):
        self.series = []
    def __call__(self, new_value):
        self.series.append(new_value)
        total = sum(self.series)
        return total/len(self.series)

ave = Averager()
print(ave(10))
print(ave(11))

# closure

def make_averager():
    series = [] # free variable
    def averager(new_value):
        series.append(new_value)
        total = sum(series)
        return total/len(series)
    return averager
ave = make_averager()
print(ave(10))
print(ave(11))
print(ave(12))

print(ave.__code__.co_varnames)
print(ave.__code__.co_freevars)

