import fibo

print("#######################################")
fibo.fib(1000)
print(fibo.fib2(500))

print(fibo.__name__)

fib = fibo.fib
fib(100)