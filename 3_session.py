# -*- coding:utf-8 -*-

import tensorflow as tf

result = tf.constant(2)

sess = tf.Session()
# 会话作为默认会话
with sess.as_default():
    # 不用显式声明 print(sess.run(result)), result就在默认会话里
    print(result.eval())

# 交互式环境下的默认会话
sess = tf.InteractiveSession()
print(result.eval())
sess.close()

# 按照配置configproto生成会话。
# 第一个参数表示GPU可以转到CPU，可移植增强，默认false。第二个参数记录每个节点安排在哪个设备上，通常取false
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess = tf.Session(config=config)
with sess.as_default():
    print(result.eval())

