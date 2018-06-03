# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: Roy
@contact: iranpeng@gmail.com
@file: train.py
@time: 2018/6/2 19:24
"""
import glob
import tensorflow as tf
import logging

from deepid import meta
from deepid.classifier import DeepIDClassifier

tf.app.flags.DEFINE_string('meta_file_path', '../../data/meta.json', 'Meta info file')
tf.app.flags.DEFINE_integer('max_steps', 300000, 'Max training steps')
tf.app.flags.DEFINE_string('model_dir', '../../model', 'Directory to save model parameters, graph and etc.')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')
tf.app.flags.DEFINE_float('drop_rate', 0.75, 'Dropout rate for training')

FLAGS = tf.app.flags.FLAGS


def parser(record, image_size):
    keys_to_features = {
        'image': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(image, shape=image_size)
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255.0)
    image = tf.multiply(tf.subtract(image, 0.5), 2)
    label = tf.cast(parsed['label'], tf.int32)
    return image, label


def dataset_input_fn(tfrecords_path):
    meta_data = meta.get_meta_data(FLAGS.meta_file_path)
    image_size = [meta_data.image_height, meta_data.image_width, meta_data.num_channels]
    filenames = glob.glob(tfrecords_path)
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(lambda record: parser(record, image_size))
    dataset = dataset.repeat(-1)
    dataset = dataset.batch(FLAGS.batch_size)
    train_iterator = dataset.make_one_shot_iterator()
    features, labels = train_iterator.get_next()
    return features, labels


def main(argv):
    meta_data = meta.get_meta_data(FLAGS.meta_file_path)
    logging.getLogger().setLevel(logging.INFO)
    train_tfrecords_path = "../../data/train/*.tfrecord"
    val_tfrecords_path = "../../data/val/*.tfrecord"
    val_num_batches = int(meta_data.num_val_examples / FLAGS.batch_size)
    classifier = DeepIDClassifier(model_dir=FLAGS.model_dir,
                                  meta_file_path=FLAGS.meta_file_path,
                                  drop_rate=FLAGS.drop_rate)
    classifier.train(lambda: dataset_input_fn(train_tfrecords_path),
                     lambda: dataset_input_fn(val_tfrecords_path),
                     val_num_batches=val_num_batches,
                     max_steps=FLAGS.max_steps)


if __name__ == '__main__':
    tf.app.run(main)
