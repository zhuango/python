from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

srng = RandomStreams(seed = 234)
rv_u = srng.uniform((2, 2))
rv_n = srng.normal((2, 2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates = True)
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

f_val0 = f()
print(f_val0)
f_val1 = f()
print(f_val1)

g_val0 = g()
print(g_val0)
g_val1 = g()
print(g_val1)

nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
print(nearly_zeros())

####Seeding Streams
rng_val = rv_u.rng.get_value(borrow=True)
rng_val.seed(89234)
rv_u.rng.set_value(rng_val, borrow=True)

###Sharing Streams Between Functions
state_after_v0 = rv_u.rng.get_value().get_state()
print(nearly_zeros())

v1 = f()
print(v1)
rng = rv_u.rng.get_value(borrow = True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow = True)

v2 = f()
print(v2)
v3 = f()
print(v3)
