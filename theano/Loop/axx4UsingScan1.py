import theano
import theano.tensor as T

k = T.iscalar("k")
A = T.vector("Aa")

results, updates = theano.scan(fn = lambda a, b, c: c,
                              sequences=[A, A],
                              outputs_info=T.ones_like(A[0]),
                              non_sequences = [10, 1000],
                              n_steps=k)
scanResult = theano.function(inputs=[A, k], outputs=results)

print(scanResult(range(10), 10))