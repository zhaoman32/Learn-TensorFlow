# -*- coding:utf-8 -*-
# 正则化 L1：wi和
# 正则化 L2：wi平方和，求导方便再乘以1/2
# 正则化 L1L2：aL1+(1-a)L2

import tensorflow as tf
from numpy.random import RandomState
weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])

'''
# 这里使用L2正则化, loss = 均方差 + 正则化因子
loss = tf.reduce_mean(tf.square(y_ - y) +
                      tf.contrib.layers.l2_regularizer(lambda)(weights)

'''

with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(weights)))
    print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(weights)))


# ——————————————————————计算一个五层神经网络的L2正则化方法——————————————————————————

def get_weights(shape, lamda):
    # 生成变量
    var = tf.Variable(tf.random_normal(shape), dtype= tf.float32)
    # add_to_collection 将L2正则化误差加入集合。第一参集合名，第二参加入集合的内容
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamda)(var))
    return var


batch_size = 8

x = tf.placeholder(tf.float32, shape=(None, 2), name='x_input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')

# 定义每一层网络节点个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)

# 这个变量维护前向传播时最深的点，开始就是输入层
cur_layer = x
# 初始化当前层的节点个数
in_dimension = layer_dimension[0]

# 通过一个循环来生成5层全连接的神经网络
for i in range(1, n_layers):
    # layer_dimension[i]为下一层的节点个数，即输出的节点个数。weight维度是in_dimension * out_dimension
    out_dimension = layer_dimension[i]
    # 生成当前层的权重，L2损失已经在函数get_weight中加进去了
    weight = get_weights([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))

    # 激活函数RELU
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 更新当前节点个数
    in_dimension = layer_dimension[i]

# 均方误差
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 把均方误差也加入losses集合
tf.add_to_collection('losses', mse_loss)

# 总的loss为losses中的所有相加
loss = tf.add_n(tf.get_collection('losses'))


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
    print(sess.run(weight))





