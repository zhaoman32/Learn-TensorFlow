# -*- coding:utf-8 -*-
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, ".\model\model.ckpt")
    # 计算元图以json形式导出
    saver.export_meta_graph(".\model\model.ckpt.meda.json", as_text=True)

"""
model.ckpt.meta 计算图的结构
model.ckpt.data-00000-of-00001 变量取值
checkpoint 模型文件列表
"""
# 重读回来
with tf.Session() as sess:
    saver.restore(sess, ".\model\model.ckpt")
    print(sess.run(result))


