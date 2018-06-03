# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: Roy
@contact: iranpeng@gmail.com
@file: convert_images_to_tfrecords.py
@time: 2018/6/2 14:04
"""
import os
import sys
import numpy as np
import tensorflow as tf

from utils import tfrecord_utils
from deepid.meta import Meta

tf.app.flags.DEFINE_string('data_dir', r'../../data',
                           'Directory to write the converted files')
FLAGS = tf.app.flags.FLAGS


def read_csv_file(csv_file):
    print("Reading csv file from:", csv_file)
    paths, labels = list(), list()
    with open(csv_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            path, label = line.strip().split()
            paths.append(path)
            labels.append(int(label))
    print("Total samples: {}, num classes: {}".format(len(labels), np.max(labels)+1))
    return paths, labels


def read_csv_pair_file(csv_file):
    print("Reading csv file from:", csv_file)
    paths_p1, paths_p2, labels = list(), list(), list()
    with open(csv_file, "r") as f:
        for line in f.readlines():
            p1, p2, label = line.strip().split()
            paths_p1.append(p1)
            paths_p2.append(p2)
            labels.append(int(label))
    print("Total samples: {}, num classes: {}".format(len(labels), np.max(labels)+1))
    return paths_p1, paths_p2, labels


def save_images_to_tfrecords(paths, labels, out_dir, prefix_filename, max_count_one_tfrecord_file=5000):
    if not paths or not labels:
        return 0
    assert(len(paths) == len(labels))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    total_count = len(labels)
    writer = None
    record_index = 0
    index = 0
    for path, label in zip(paths, labels):
        if index % max_count_one_tfrecord_file == 0:
            if writer:
                writer.close()
            tfrecord_full_path = os.path.join(out_dir, "".join([prefix_filename, "_", str(record_index), ".tfrecord"]))
            sys.stdout.write("\nSave tfrecord to file: %s\n" % tfrecord_full_path)
            writer = tf.python_io.TFRecordWriter(tfrecord_full_path)
            record_index += 1
        image_bytes = tfrecord_utils.read_image_file(path)
        example = tfrecord_utils.create_image_example(image_bytes, label)
        writer.write(example.SerializeToString())
        index += 1
        sys.stdout.write("\rProcess: %d/%d %.2f%%" % (index, total_count, index*100/float(total_count)))
    if writer:
        writer.close()

    return index


def create_tfrecords_meta_file(num_train_examples, num_val_examples,
                               num_test_examples, num_classes,
                               image_width, image_height, num_channels,
                               path_to_tfrecords_meta_file):
    print('Saving meta file to %s...' % path_to_tfrecords_meta_file)
    meta = Meta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    meta.num_classes = num_classes
    meta.image_width = image_width
    meta.image_height = image_height
    meta.num_channels = num_channels
    meta.save(path_to_tfrecords_meta_file)


def save_test_images_to_tfrecords(paths_p1, paths_p2,
                                  labels, out_dir,
                                  prefix_filename, max_count_one_tfrecord_file=5000):
    if not paths_p1 or not paths_p2 or not labels:
        return 0
    assert(len(paths_p1) == len(paths_p2) == len(labels))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    total_count = len(labels)
    writer = None
    record_index = 0
    index = 0
    for path1, path2, label in zip(paths_p1, paths_p2, labels):
        if index % max_count_one_tfrecord_file == 0:
            if writer:
                writer.close()
            tfrecord_full_path = os.path.join(out_dir, "".join([prefix_filename, "_", str(record_index), ".tfrecord"]))
            sys.stdout.write("\nSave tfrecord to file: %s\n" % tfrecord_full_path)
            writer = tf.python_io.TFRecordWriter(tfrecord_full_path)
            record_index += 1
        image_bytes1 = tfrecord_utils.read_image_file(path1)
        image_bytes2 = tfrecord_utils.read_image_file(path2)
        example = tfrecord_utils.create_image_example2(image_bytes1, image_bytes2, label)
        writer.write(example.SerializeToString())
        index += 1
        sys.stdout.write("\rProcess: %d/%d %.2f%%" % (index, total_count, index*100/float(total_count)))
    if writer:
        writer.close()


if __name__ == '__main__':
    train_dataset_csv_file = os.path.join(FLAGS.data_dir, 'train_set.csv')
    val_dataset_csv_file = os.path.join(FLAGS.data_dir, 'valid_set.csv')
    test_dataset_csv_file = os.path.join(FLAGS.data_dir, 'test_set.csv')
    tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')
    image_width = 47
    image_height = 55
    num_channels = 3

    train_saved_dir = os.path.join(FLAGS.data_dir, 'train')
    val_saved_dir = os.path.join(FLAGS.data_dir, 'val')
    test_saved_dir = os.path.join(FLAGS.data_dir, 'test')
    saved_prefix_filename = "youtube_face"
    train_paths, train_labels = read_csv_file(train_dataset_csv_file)
    val_paths, val_labels = read_csv_file(val_dataset_csv_file)
    test_paths_p1, test_paths_p2, test_labels = read_csv_pair_file(test_dataset_csv_file)

    processed_cout = save_images_to_tfrecords(train_paths, train_labels, train_saved_dir, saved_prefix_filename)
    print("\nProcessed count:", processed_cout)
    processed_cout = save_images_to_tfrecords(val_paths, val_labels, val_saved_dir, saved_prefix_filename)
    print("\nProcessed count:", processed_cout)
    processed_cout = save_test_images_to_tfrecords(test_paths_p1, test_paths_p2, test_labels, test_saved_dir, saved_prefix_filename)
    print("\nProcessed count:", processed_cout)
    create_tfrecords_meta_file(num_train_examples=len(train_labels), num_val_examples=len(val_labels),
                               num_test_examples=len(test_labels), num_classes=np.max(train_labels)+1,
                               image_width=image_width, image_height=image_height, num_channels=num_channels,
                               path_to_tfrecords_meta_file=tfrecords_meta_file)

