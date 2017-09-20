def make_incrementor(n):
    return lambda x: x + n
f = make_incrementor(42)
print("f(0) = ", f(0))
print("f(1) = ", f(1))

pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
pairs.sort(key = lambda pair: pair[1])
print("pairs = ", pairs)

print("======Fluent python======")
fruits = ['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana']
reverse_sort = sorted(fruits, key=lambda word: word[::-1])
print(reverse_sort)
