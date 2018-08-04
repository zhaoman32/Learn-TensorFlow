# -*- coding:utf-8 -*-

import tensorflow as tf

# 前向传播
x = [[0.7, 0.9]]
w1 = [[0.2, 0.1, 0.4], [0.3, -0.5, 0.2]]
w2 = [[0.6], [0.1], [-0.2]]
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


with tf.Session() as sess:
    print(sess.run(y))


# 随机矩阵生成

w1 = tf.random_normal([2, 3], stddev=2)
w2 = tf.zeros([3, 3])
w3 = tf.fill([5, 5], 8)
w4 = w3 * 2

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(w1))
    print(sess.run(w2))
    print(sess.run(w3))
    print(sess.run(w4))


