#!/usr/bin/python3

callableOrNot = [callable(obj) for obj in (abs, str, 13)]
print(callableOrNot)

import random
class BingoCage:
    def __init__(self, items):
        self._items = list(items)
        random.shuffle(self._items)
    def pick(self):
        try:
            return self._items.pop()
        except IndexError:
            raise LookupError('pick from empty BingoCage')
    def __call__(self):
        return self.pick()
bingo = BingoCage(range(3))
print(bingo.pick())
pickk = bingo()
print(pickk)
print(callable(bingo))
