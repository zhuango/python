#!/usr/bin/python3
import collections

# namedtuple can be used to build classes
# of objects that are just bundles of attributes with no custom methods, 
# like a database record.
Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list("JQKA")
    suits = "spades diamonds clubs hearts".split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                                       for rank in self.ranks]
    def __len__(self):
        return len(self._cards)
    
    def __getitem__(self, position):
        return self._cards[position]

from random import choice
# 1. The users of your classes don’t have to memorize arbitrary method names for stan‐
# dard operations (“How to get the number of items? Is it .size() .length() or what?”)
# 2. It’s easier to benefit from the rich Python standard library and avoid reinventing
# the wheel, like the random.choice function.
if __name__ == "__main__":
    card = Card('7', 'diamonds')
    print(card)

    fd = FrenchDeck()
    print(len(fd))
    print(fd[1])
    print(choice(fd))
    print(choice(fd))

    print(fd[:3])
    print(fd[12:-1:13])

    for card in fd:
        print(card)
    for card in reversed(fd):
        print(card)
# If a collection has no __contains__ method, the in operator
# does a sequential scan. Case in point: in works with our FrenchDeck class because it is
# iterable.
    if Card('Q', 'hearts') in fd:
        print(True)
    if Card('7', 'beasts') in fd:
        print(True)