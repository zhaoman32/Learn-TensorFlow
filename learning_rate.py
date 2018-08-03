# -*- coding:utf-8 -*-
import tensorflow as tf

# decayed_learning_rate 每一轮优化时的学习率
# learn_rate 初始学习率
# decay_rate 衰减系数
# global_step 迭代轮数
# decay_steps 衰减速度
# decayed_learning_rate = learn_rate * decay_rate ^ (global_step / decay_steps)

global_step = tf.variable(0)
learning_rate = tf.train.exponential_decay(learning_rate=0.1,
                                           global_step=global_step,
                                           decay_steps=100,
                                           decay_rate=0.96,
                                           staircase=True
                                           )

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss , global_step=global_step)
# 初始学习率0.1，每迭代一轮乘以0.96
