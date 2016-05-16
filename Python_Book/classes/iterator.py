class Reverse:
    """Iterator for looping over a sequence backwards."""
    def __init__(self, data):
        self.data  = data
        self.index = len(data)
        
    def __iter__(self):
        return self
    def __next__(self):# python3
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]
    def next(self):# python2
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]


s = 'abc'
it = iter(s)
for char in it:
    print(char)
print('###############################')
rev = Reverse('spam')
#iter(rev)
for char in rev:
    print(char)