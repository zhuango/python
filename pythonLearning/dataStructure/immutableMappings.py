#!/usr/bin/python3

from types import MappingProxyType
d = {1:'A'}
# a view of d.
d_proxy = MappingProxyType(d)
print(d_proxy)
# Error! changing dict through the proxy is not allowed.
d_proxy[1] = 'sdfsd'
