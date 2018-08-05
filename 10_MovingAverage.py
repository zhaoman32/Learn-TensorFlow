# -*- coding:utf-8 -*-

import tensorflow as tf

"""
滑动平均模型，它可以使得模型在测试数据上更健壮.
在使用随机梯度下降算法训练神经网络时，通过滑动平均模型可以在很多的应用中在一定程度上提高最终模型在测试数据上的表现。
其实滑动平均模型，主要是通过控制衰减率来控制参数更新前后之间的差距，从而达到减缓参数的变化值
（如，参数更新前是5，更新后的值是4，通过滑动平均模型之后，参数的值会在4到5之间）
如果参数更新前后的值保持不变，通过滑动平均模型之后，参数的值仍然保持不变。
计算公式：shadow_variable = decay * shadow_variable + (1-decay) * variable

"""
# 定义一个变量用于计算滑动平均，变量的初始值为0，变量的类型必须是实数
v1 = tf.Variable(5, dtype=tf.float32)
# 定义一个迭代轮数的变量，动态控制衰减率,并设置为不可训练
step = tf.Variable(10, trainable=False)
# 定义一个滑动平均类，初始化衰减率为0.99和衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义每次滑动平均所更新的列表
maintain_average_op = ema.apply([v1])
# 初始化上下文会话
with tf.Session() as sess:
    # 初始化所有变量
    init = tf.global_variables_initializer()
    sess.run(init)
    # 更新v1的滑动平均值
    '''
    衰减率为min(0.99,(1+step)/(10+step)=0.1}=0.1
    '''
    sess.run(maintain_average_op)
    # [5.0, 5.0]
    print(sess.run([v1, ema.average(v1)]))
    sess.run(tf.assign(v1, 4))
    sess.run(maintain_average_op)
    # [4.0, 4.5500002],5*(11/20) + 4*(9/20)
    print(sess.run([v1, ema.average(v1)]))