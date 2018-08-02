# -*- coding:utf-8 -*-
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

x = tf.placeholder(tf.float32, shape=(None, 2), name='x_input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 多种损失函数可选

# 1.交叉熵
# loss = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# 2.softmax处理后的交叉熵(存疑)
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=y_))
# 3.均方误差
# loss = tf.reduce_mean(tf.square(y_ - y))
# 4.自定义loss ,select = where
loss = tf.reduce_mean(tf.where(tf.greater(y, y_), (y - y_) * 1, (y_ - y) * 10))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 定义数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[x1 + x2 + rdm.rand() / 10 - 0.05] for (x1, x2) in X]

# 创建会话
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 10000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练，这里防止超过dataset_size处理
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        # 每隔1000轮我们输出一下
        if i % 1000 == 0:
            total_cross_entropy = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
            # 这个交叉熵会越来越小

    print(sess.run(w1))
