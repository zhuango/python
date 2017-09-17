class Cheese:
    def __init__(self, kind):
        self.kind = kind
    def __repr__(self):
        return 'Cheese(%r)' % self.kind

import weakref
stock = weakref.WeakValueDictionary()
catalog = [Cheese('Red Leicester'), Cheese('Tilsit'), Cheese('Brie'), Cheese('Parmesan')]

for cheese in catalog:
    stock[cheese.kind] = cheese

result0 = sorted(stock.keys())
print(result0)

del catalog
result1 = sorted(stock.keys())
print(result1)

del cheese
result2 = sorted(stock.keys())
print(result2)
