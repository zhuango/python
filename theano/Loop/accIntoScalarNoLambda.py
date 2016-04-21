import numpy as np
import theano
import theano.tensor as T 
up_to = T.iscalar("up_to")

def accumulate_by_adding(arange_val, sum_to_date):
    return sum_to_date + arange_val

seq = T.arange(up_to)

outputs_info = T.as_tensor_variable(np.asarray(0, seq.dtype))
scan_result, scan_updates = theano.scan(fn=accumulate_by_adding,
                                        outputs_info = outputs_info,
                                        sequences = seq)
triangular_sequence = theano.function(inputs = [up_to], outputs = scan_result)
some_num = 15
print(triangular_sequence(some_num))
print([n*(n + 1) // 2 for n in range(some_num)])