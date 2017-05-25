#!/usr/bin/python3
# The collections.namedtuple function is a factory that produces subclasses of tuple
# enhanced with field names and a class name — which helps debugging

from collections import namedtuple
# Two parameters are required to create a named tuple: a class name and a list of
# field names, which can be given as an iterable of strings or as a single spacedelimited string.
City = namedtuple('City', 'name country population coordinates')
tokyo = City("Tokyo", "JP",  36.933, (35.689722, 139.691667))
print(tokyo)
print(tokyo.population)
print(tokyo.coordinates)
print(tokyo[1])

# attributes of named tuple
print("________attributes of named tuple________")
print(City._fields)
LatLong = namedtuple("LatLong", "lat long")
delhi_data = ('Delhi NCR', 'IN', 21.935, LatLong(28.613889, 77.208889))
# _make() lets you instantiate a named tuple from an iterable; 
# City(*delhi_data) would do the same.
delhi = City._make(delhi_data)
print(delhi)
# _asdict() returns a collections.OrderedDict built from the named tuple
# instance. That can be used to produce a nice display of city data.
d = delhi._asdict()
print(d)

for key, value in delhi._asdict().items():
    print(key, ":", value)
