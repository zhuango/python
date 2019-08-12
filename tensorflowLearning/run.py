import numpy as np
import tensorflow as tf
from tensorflow import nn
b = tf.random_normal([1],seed = 1234)
#arr = np.array([1, 5.5, 3, 15, 20])
myRandom = np.random.RandomState(2333)

from my_layer_norm import *

arr = myRandom.uniform(-1, 1, (2, 3, 4))
name = "bert"
tensor = tf.contrib.layers.layer_norm(inputs=arr, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
tensor2 = my_layer_norm(inputs=arr, begin_norm_axis=-1, begin_params_axis=-1, scope="bert2")
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(tensor))
    print("SSSSSSSSSSSSSSSSSSSSS")
    print(sess.run(tensor2))
    # saver=tf.train.Saver(tf.global_variables(),max_to_keep=10)
    # print("model: ",saver.save(sess,'modle.ckpt'))
    print(tf.global_variables())
    saver=tf.train.Saver(tf.global_variables())
    saver.restore(sess, 'modle.ckpt')
    print(sess.run(tensor))
    print("SSSSSSSSSSSSSSSSSSSSS")
    print(sess.run(tensor2))
