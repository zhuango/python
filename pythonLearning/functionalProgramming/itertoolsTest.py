#!/usr/bin/python3
import operator
import itertools
import sys

def myAccumulate(iterable, func = operator.add):
    'Return running totals'
    #  accumulate([1, 2,3 ,4 5]) --> 1 3 6 10 15
    #  accumulate([1, 2,3, 4, 5], operator.mul) --> 1 2 6 34 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return 
    yield total
    for element in it:
        total= func(total, element)
        yield total

for result in myAccumulate([1, 2, 3, 4, 5]):
    sys.stdout.write(str(result) + " ")
print()

for result in myAccumulate([1, 2, 3, 4, 5], operator.mul):
    sys.stdout.write(str(result) + " ")
print()    

for result in itertools.accumulate([1, 2, 3, 4, 5]):
    sys.stdout.write(str(result) + " ")
print()

for result in itertools.accumulate([1, 2, 3, 4, 5], operator.mul):
    sys.stdout.write(str(result) + " ")
print()

for result in itertools.accumulate([231, 234, 1, 23, 12312], max):
    sys.stdout.write(str(result) + " ")
print()

cashflows = [1000, -90, -90, -90, -90]
results = list(itertools.accumulate(cashflows, lambda bal, pmt: bal*1.05 + pmt))
print(results)

logistic_map = lambda x, _: r * x * (1 - x)
r = 3.8
x0 = 0.4

inputs = itertools.repeat(x0, 36)
for result in itertools.accumulate(inputs, logistic_map):
    sys.stdout.write(format(result, '.2f') + " ")
print()

for res in itertools.chain([1, 2, 3, 4], [5, 6, 7, 8]):
    sys.stdout.write(str(res) + " ")
print()

for res in itertools.chain.from_iterable(["dssf", "sfs"]):
    sys.stdout.write(str(res) + " ")
print()

for res in itertools.compress("ABCDEFGHIJKLMNOPQRST", [1,0,1,0,1,1]):
    sys.stdout.write(str(res) + " ")
print()

for number in itertools.count(0, 0.5):
    if(number > 100): break
    sys.stdout.write(str(number) + " ")
print()

cycle=0
for elem in itertools.cycle("liu"):
    sys.stdout.write(str(elem) + " ")
    cycle += 1
    if(cycle > 3 * 10): break
print()

testData = [1, -2, 321, -2, -213, 2, 123]
# drop element from start to the index which first makes predicate function flase.
for elem in itertools.dropwhile(lambda x : x > 0, testData):
    sys.stdout.write(str(elem) + " ")
print()

print(list(itertools.dropwhile(lambda x : x == ' ', "     liuzhuang  $")))