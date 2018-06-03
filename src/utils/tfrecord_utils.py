# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: Roy
@contact: iranpeng@gmail.com
@file: tfrecord_utils.py
@time: 2018/6/2 14:11
"""
import os
import tensorflow as tf
from PIL import Image


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def read_image_file(file_path, resize=None, grayscale=False):
    if not os.path.exists(file_path):
        print("No such file or directory: %s" % file_path)
        return None
    image = Image.open(file_path)
    if grayscale:
        image = image.convert('LA')
    if resize:
        image = image.resize(resize)

    image_bytes = image.tobytes()
    return image_bytes


def create_image_example(image_bytes, label):
    if not image_bytes:
        return None
    feature = {
        "image": bytes_feature(image_bytes),
        "label": int64_feature(label),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def create_image_example2(image_bytes1, image_bytes2, label):
    if not image_bytes1 or not image_bytes2:
        return None
    feature = {
        "image1": bytes_feature(image_bytes1),
        "image2": bytes_feature(image_bytes2),
        "label": int64_feature(label),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

