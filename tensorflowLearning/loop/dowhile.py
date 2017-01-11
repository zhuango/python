import numpy as np
import tensorflow as tf
i = tf.constant(0)
l = tf.Variable(np.zeros((4, 1), dtype=np.float32), dtype=tf.float32)

array = tf.Variable(tf.random_normal([4, 4]))
#temp = tf.transpose(tf.gather(array,[i]))
print(l)
def cond(i,l):
   return i < 4

def body(i, l):                                               
    temp = tf.transpose(tf.gather(array,[i]))
    l = tf.concat(0, [l, [temp]])
    return i+1, l

index, list_vals = tf.while_loop(cond, body, [i, l], shape_invariants=[i.get_shape(), tf.TensorShape([4, 1])])
sess = tf.Session()
sess.run(tf.initialize_all_variables())
ii, a, result= sess.run([index, array, list_vals])
print(ii)
print(a)
print(result)