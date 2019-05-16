import html
from functools import singledispatch
from collections import abc
import numbers

@singledispatch
def htmlize(obj):
    content = html.escape(repr(obj))
    return '<pre>{}</pre>'.format(content)

@htmlize.register(str)
def _(text):
    content = html.escape(text).replace("\n", "<br>\n")
    return '<pre>{}</pre>'.format(content)

@htmlize.register(numbers.Integral)
def _(n):
    return "<pre>{0} (0x{0:x})</pre>".format(n)

@htmlize.register(tuple)
@htmlize.register(abc.MutableSequence)
def _(seq):
    inner = "</lr>\n<li>".join(htmlize(item) for item in seq)
    return "<ul>\n<li>" + inner + "</li>\n</ul>"



def ifelseHtmlize(obj):
    if isinstance(obj, str):
        content = html.escape(repr(obj))
        content = '<pre>{}</pre>'.format(content)
    elif isinstance(obj, numbers.Integral):
        content = "<pre>{0} (0x{0:x})</pre>".format(obj)
    
    return content
    

if __name__ == "__main__":
    s = htmlize("sssss")
    print(s)

    # print(ifelseHtmlize("DSS"))
    # print(ifelseHtmlize(123))