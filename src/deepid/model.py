# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: Roy
@contact: iranpeng@gmail.com
@file: model.py
@time: 2018/6/2 19:24
"""

import tensorflow as tf

_DEEP_ID_SIZE = 160


def inference(X, drop_rate=1.0):
    with tf.variable_scope("Conv1"):
        conv1 = tf.layers.conv2d(X, filters=20, kernel_size=[4, 4])
        relu1 = tf.nn.relu(conv1)
    with tf.variable_scope("Pool1"):
        pool1 = tf.layers.max_pooling2d(relu1, pool_size=[2, 2], strides=2)
        dropout1 = tf.layers.dropout(pool1, rate=drop_rate)
        hidden1 = dropout1

    with tf.variable_scope("Conv2"):
        conv2 = tf.layers.conv2d(hidden1, filters=40, kernel_size=[3, 3])
        relu2 = tf.nn.relu(conv2)
    with tf.variable_scope("Pool2"):
        pool2 = tf.layers.max_pooling2d(relu2, pool_size=[2, 2], strides=2)
        dropout2 = tf.layers.dropout(pool2, rate=drop_rate)
        hidden2 = dropout2

    with tf.variable_scope("Conv3"):
        conv3 = tf.layers.conv2d(hidden2, filters=60, kernel_size=[3, 3])
        relu3 = tf.nn.relu(conv3)
    with tf.variable_scope("Pool3"):
        pool3 = tf.layers.max_pooling2d(relu3, pool_size=[2, 2], strides=2)
        dropout3 = tf.layers.dropout(pool3, rate=drop_rate)
        hidden3 = dropout3

    with tf.variable_scope("Conv4"):
        conv4 = tf.layers.conv2d(hidden3, filters=80, kernel_size=[2, 2])

    with tf.variable_scope("DeepID"):
        fc11 = tf.layers.dense(tf.layers.flatten(hidden3), units=_DEEP_ID_SIZE, name='fc11')
        fc12 = tf.layers.dense(tf.layers.flatten(conv4), units=_DEEP_ID_SIZE, name='fc12')
        deepid = tf.nn.relu(tf.add(fc11, fc12), name='deepid')
    print(relu1.name, relu1.get_shape())
    print(pool1.name, pool1.get_shape())
    print(relu2.name, relu2.get_shape())
    print(pool2.name, pool2.get_shape())
    print(relu3.name, relu3.get_shape())
    print(pool3.name, pool3.get_shape())
    print(conv4.name, conv4.get_shape())
    print(deepid.name, deepid.get_shape())
    return deepid

