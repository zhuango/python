#/usr/bin/python3
import math

# entorpy = -sum_on_x( p(x) * log2( p(x) ) )
def entropyDiscrete(variableCount, probs):
    entropy = 0.0
    for i in xrange(0, variableCount):
        entropy -= probs[i] * math.log(probs[i], 2.0)
    return entropy