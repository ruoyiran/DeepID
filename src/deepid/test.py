# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: Roy
@contact: iranpeng@gmail.com
@file: test.py
@time: 2018/6/3 1:02
"""
import glob

import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cosine

from deepid import meta
from deepid.classifier import DeepIDClassifier

tf.app.flags.DEFINE_string('meta_file_path', '../../data/meta.json', 'Meta info file')
tf.app.flags.DEFINE_string('model_dir', '../../model', 'Directory to save model parameters, graph and etc.')

FLAGS = tf.app.flags.FLAGS


def decode_and_preprocess_image(raw_image, image_size):
    image = tf.decode_raw(raw_image, tf.uint8)
    image = tf.reshape(image, shape=image_size)
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255.0)
    image = tf.multiply(tf.subtract(image, 0.5), 2)
    return image


def parser(record, image_size):
    keys_to_features = {
        'image1': tf.FixedLenFeature((), tf.string),
        'image2': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image1 = decode_and_preprocess_image(parsed['image1'], image_size)
    image2 = decode_and_preprocess_image(parsed['image2'], image_size)
    label = tf.cast(parsed['label'], tf.int32)
    return image1, image2, label


def dataset_input_fn(tfrecords_path):
    meta_data = meta.get_meta_data(FLAGS.meta_file_path)
    image_size = [meta_data.image_height, meta_data.image_width, meta_data.num_channels]
    filenames = glob.glob(tfrecords_path)
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(lambda record: parser(record, image_size))
    dataset = dataset.repeat(1)
    dataset = dataset.batch(meta_data.num_test_examples)
    train_iterator = dataset.make_one_shot_iterator()
    return train_iterator.get_next()


def main(argv):
    test_tfrecords_path = "../../data/test/*.tfrecord"
    classifier = DeepIDClassifier(model_dir=FLAGS.model_dir, meta_file_path=FLAGS.meta_file_path,
                                  model_path="../../model/best_model.ckpt")
    image1_batch, image2_batch, label_batch = dataset_input_fn(test_tfrecords_path)
    images1, images2, labels = classifier.session.run([image1_batch, image2_batch, label_batch])
    deepid_features1 = classifier.extract_feature(images1)
    deepid_features2 = classifier.extract_feature(images2)

    pre_y = np.array([cosine(x, y) for x, y in zip(deepid_features1, deepid_features2)])

    def part_mean(x, mask):
        z = x * mask
        return float(np.sum(z) / np.count_nonzero(z))

    true_mean = part_mean(pre_y, labels)
    false_mean = part_mean(pre_y, 1 - labels)
    print("total samples:", len(labels))
    print("true_mean:", true_mean)
    print("false_mean:", false_mean)
    accuracy = np.mean((pre_y < (true_mean + false_mean) / 2) == labels.astype(bool))
    print("accuracy: %.2f%%" % (accuracy*100))


if __name__ == '__main__':
    tf.app.run(main)
