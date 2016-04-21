import theano 
import theano.tensor as T
import numpy

# seen as a list of location.
locations = T.imatrix("locations")
values = T.vector("values")
output_model = T.matrix("output_model")

def set_value_at_position(a_location, a_value, output_model):
    zeros = T.zeros_like(output_model)
    zeros_subtensor = zeros[a_location[0], a_location[1]]
    return T.set_subtensor(zeros_subtensor, a_value)

result, updates = theano.scan(fn=set_value_at_position,
                              outputs_info = None,
                              sequences = [locations, values],
                              non_sequences=output_model)
assign_values_at_position=theano.function(inputs=[locations, values, output_model], outputs=result)

test_locations = numpy.asarray([[1, 1], [2, 3]], dtype=numpy.int32)
test_values = numpy.asarray([42, 50], dtype=numpy.float32)
test_output_model = numpy.zeros((5, 5), dtype = numpy.float32)
print(assign_values_at_position(test_locations, test_values, test_output_model))