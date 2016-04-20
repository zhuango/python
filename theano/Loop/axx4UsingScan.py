import theano
import theano.tensor as T
import numpy 

k = T.iscalar("k")
A = T.vector("A")

# Symbolic description of the result
result, updates = theano.scan(fn=lambda prior_resul, A: prior_resul * A,
                              outputs_info=T.ones_like(A), #content includes all 1s like the shape of A
                              non_sequences=A,
                              n_steps=k)

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
# final_result = result[-1]

# compiled function that returns A**k
power = theano.function(inputs=[A,k], outputs=result)

print(power(range(10),2))
print(power(range(10),4))