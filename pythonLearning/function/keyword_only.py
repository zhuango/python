def tag(name, *content, cls=None, **attrs):
    """Genenrate one or more HTML tags"""
    if cls is not None:
        attrs['class'] = cls
    if attrs:
        attr_str = ''.join(' %s="%s"' % (attr, value)
                            for attr, value
                            in sorted(attrs.items()))
    else:
        attr_str = ''
    if content:
        return '\n'.join('<%s%s>%s</%s>' %
                        (name, attr_str, c, name) for c in content)
    else:
        return '<%s%s />' % (name, attr_str)

print(tag('br'))
print(tag('p', 'hello'))
print(tag('p', 'hello', 'world'))
print(tag('p', 'hello', id=33))
print(tag('p', 'hello', 'world', cls='sidebar'))
my_tag = {'name':'img', 'title':'Sunset Boulevard', 'src':'sunset.jpg', 'cls':'framed'}
print(tag(**my_tag))

#If you don’t want to support variable positional arguments but still want keyword-only arguments, put a * by itself in the signature,
def f(a, *, b):
    return a, b
result = f(1, b=10)
print(result)
