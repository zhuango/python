import theano
import theano.tensor as T
import time

t0 = time.time()

#s(x) = 1 / (1 + exp(-x)) = (1 + tanh(x / 2)) / 2
x = T.matrix('x')
s = 1 / (1 + T.exp(-x))#formula

logistic = theano.function([x], s)
print (logistic([[0, 1], [-1, -2]]))

s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = theano.function([x], s2)
print(logistic2([[0, 1], [-1, -2]]))

#Computing More than one Thing at the Same Time
a, b = T.dmatrices('a', 'b')

diff = a - b
abs_diff = abs(diff)
diff_squared = diff ** 2

f = theano.function([a, b], [diff, abs_diff, diff_squared])
print(
f(
    [
        [1, 1],
        [1, 1],
    ],
    [
        [0, 1],
        [2, 3],
    ]
)
)

#Setting a Default Value for an Argument
from theano import In
from theano import function
x, y = T.dscalars('x', 'y')
z = x + y
f = function([x, In(y, value = 1)], z)
print(f(33))

x, y, w = T.scalars('x', 'yName', 'w')
z = (x + y) * w
f = function([x, In(y, value = 1), In(w, value = 2, name = 'zhuangliu')], z)
print(f(33))

print(f(33, 2))
print(f(33, 0, 1))
print(f(33, zhuangliu = 100))
print(f(33, zhuangliu = 1, yName = 0))

#Using Shared Variables
from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates = [(state, state + inc)])

print(state.get_value())
print(accumulator(1))
print(state.get_value())
accumulator(300)
print(state.get_value())
state.set_value(-1)
print(accumulator(3))
print(state.get_value)

t1 = time.time()
print("took %f seconds" % (t1 - t0))

fn_of_state = state * 2 + inc
foo = T.scalar(dtype = state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print(skip_shared(1, 3))
print(state.get_value())

#the given parameter
print("############################################")
fn_of_state = state * 2 + inc
foo= T.scalar(dtype = state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print(skip_shared(1, 3))
print(state.get_value())

#Using Random Numbers

