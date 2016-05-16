def reverse(data):
    for index in range(len(data) - 1, -1, -1):
        yield data[index]
for char in reverse('zhuangliu'):
    print(char)
    
print(i * i for i in range(10))

X = [10, 20, 30]
Y = [7, 5, 3]
print(sum(x * y for x, y in zip(X,Y)))
