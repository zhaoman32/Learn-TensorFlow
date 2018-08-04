# -*- coding:utf-8 -*-

import tensorflow as tf
# placeholder 存放数据，内存只占一份，数据可以变化
x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
w1 = tf.random_normal([2, 3], stddev=1)
w2 = tf.random_normal([3, 1], stddev=1)

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    tf.global_variables_initializer()
    # 运行时把数据feed进去
    print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
    print(sess.run(y, feed_dict={x: [[0.1, 0.9]]}))

