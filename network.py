# -*- coding:utf-8 -*-
import tensorflow as tf
from numpy.random import RandomState

# 定义一个batch的数据大小
batch_size = 8

# 定义神经网络的参数 这里注意要求tf.variable,这样w作为变量才能优化
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x_input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')

# 前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数 交叉熵
# reduce_mean 计算张量的各个维度上的元素的平均值。
# clip_by_value将tensor进行范围限制的函数。 1e-10表示1*10的负十次方
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# 学习率
learning_rate = 0.001
# 优化方法最小化损失函数，采用的优化器是AdamOptimizer
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 定义数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)  # dataset_size*2维
# x1+x2<1为正例1，否则负例0
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建会话
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("优化前随机生成的w1，w2")
    print(sess.run(w1))
    print(sess.run(w2))

    # ----------------------------进行优化------------------------------

    # 训练轮数
    STEPS = 10000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练，这里防止超过dataset_size处理
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        # 每隔1000轮我们输出一下
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
            # 这个交叉熵会越来越小

    print(sess.run(w1))
    print(sess.run(w2))