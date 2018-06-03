# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: Roy
@contact: iranpeng@gmail.com
@file: classifier.py
@time: 2018/6/2 22:34
"""
import atexit
import os

import tensorflow as tf
import logging

from deepid import meta, model


class DeepIDClassifier(object):
    _BEST_MODEL_NAME = "best_model.ckpt"

    def __init__(self, model_dir, meta_file_path, drop_rate=1.0, restore_model=True, model_path=None):
        atexit.register(self.close)
        self._model_dir = model_dir
        self.session = tf.Session()
        self._model_restored = False
        self._min_learning_rate = 1e-6
        self._max_learning_rate = 1e-4
        self._learning_rate_decay_steps = 100000
        self._drop_rate = drop_rate
        self._build_network(meta_file_path)
        self._init_summaries()
        self._saver = tf.train.Saver()
        if restore_model:
            self._restore_model(model_path)

    def _get_variable(self, init_value, name=None, dtype=tf.float32):
        return tf.Variable(init_value, trainable=False, name=name, dtype=dtype)

    def _build_network(self, meta_file_path):
        meta_data = meta.get_meta_data(meta_file_path)
        self.best_val_acc = self._get_variable(0.0, name='best-val-acc')
        self.learning_rate = self._get_variable(self._max_learning_rate, name='learning-rate')
        self.drop_rate_placehoder = tf.placeholder(dtype=tf.float32, name="drop-rate")
        with tf.variable_scope("Input"):
            self.input_x = tf.placeholder(dtype=tf.float32,
                                          shape=(None, meta_data.image_height,
                                                 meta_data.image_width,
                                                 meta_data.num_channels),
                                          name='input-x')
            self.input_y = tf.placeholder(dtype=tf.int32, shape=(None,), name='input-y')
            onehot_labels = tf.one_hot(self.input_y, depth=meta_data.num_classes)
        self.deepid = model.inference(self.input_x, self.drop_rate_placehoder)

        with tf.variable_scope("Softmax"):
            self.logits = tf.layers.dense(self.deepid, meta_data.num_classes, name="logits")
            self.prediction = tf.nn.softmax(self.logits)

        with tf.variable_scope("Loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=onehot_labels)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.variable_scope("Accuracy"):
            predict_labels = tf.argmax(self.prediction, axis=1, output_type=tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_labels, self.input_y), dtype=tf.float32))

        with tf.variable_scope("Trainer"):
            self.global_step = tf.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def _init_summaries(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        self._train_writer = tf.summary.FileWriter(os.path.join(self._model_dir, "train"), graph=self.session.graph)
        self._val_writer = tf.summary.FileWriter(os.path.join(self._model_dir, "val"), graph=self.session.graph)
        self._merged_summary = tf.summary.merge_all()

    def _initialize_variables(self):
        self.session.run(tf.global_variables_initializer())

    def _restore_model(self, model_path):
        model_checkpoint_path = None
        if model_path is None:
            ckpt = tf.train.get_checkpoint_state(self._model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                model_checkpoint_path = ckpt.model_checkpoint_path
        else:
            model_checkpoint_path = model_path
        if model_checkpoint_path:
            logging.info("Restoring model from '%s'", model_checkpoint_path)
            self._saver.restore(self.session, model_checkpoint_path)
            self._model_restored = True
            logging.info("Restore success. current model best val acc: %.2f%%", self.get_best_val_acc() * 100)

    def _save_model(self, model_name=None, step=None):
        if model_name:
            model_path = os.path.join(self._model_dir, model_name)
        else:
            model_path = os.path.join(self._model_dir, 'model.ckpt')
        self._protect_best_model()
        saved_model_path = self._saver.save(self.session, model_path, global_step=step)
        logging.info("Model have been saved to %s", saved_model_path)

    def train(self, train_input_fn, val_input_fn=None, val_num_batches=1, max_steps=10000):
        train_image_batch, train_label_batch = train_input_fn()
        val_image_batch, val_label_batch = None, None
        if val_input_fn:
            val_image_batch, val_label_batch = val_input_fn()
        show_loss_steps = 100
        show_acc_steps = 100
        save_check_point_steps = 100
        final_loss = 0
        step = 0
        if not self._model_restored:
            self._initialize_variables()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
        for epoch in range(max_steps+1):
            loss_val, step = self._training(train_image_batch, train_label_batch, self._train_writer)
            lr = self.session.run(self.learning_rate)
            if lr > self._min_learning_rate and step > 0 and step % self._learning_rate_decay_steps == 0:
                self._set_learning_rate(lr/10.0)
            if step % show_loss_steps == 0:
                lr = self.session.run(self.learning_rate)
                logging.info("global_step: %d, learning_rate: %f, loss: %.5f", step, lr, loss_val)
            if step % show_acc_steps == 0:
                train_acc = self._get_accuracy(train_image_batch, train_label_batch, self._train_writer)
                if val_image_batch is not None:
                    val_acc = self._evaluate_valid_accuracy(val_image_batch, val_label_batch, val_num_batches)
                    logging.info("global_step: %d, train acc: %.2f%%, val acc: %.2f%%, best val acc: %.2f%%",
                                 step, train_acc * 100, val_acc * 100, self.get_best_val_acc() * 100)
                else:
                    logging.info("global_step: %d, train acc: %.2f%%",
                                 step, train_acc * 100)
            if step > 0 and step % save_check_point_steps == 0:
                self._save_model(step=step)
            final_loss = loss_val
        self._save_model(step=step)
        logging.info("Loss for final step: %d, loss: %.5f, best acc: %.2f%%", step, final_loss, self.get_best_val_acc() * 100)
        coord.request_stop()
        coord.join(threads)
    
    def extract_feature(self, input_x):
        feature = self.session.run(self.deepid, feed_dict={
            self.input_x: input_x
        })
        return feature

    def predict(self, input_x):
        predictions = self.session.run(tf.argmax(self.prediction, 1), feed_dict={
            self.input_x: input_x
        })
        return predictions

    def close(self):
        if self.session:
            self.session.close()
            self.session = None

    def _get_feed_dict(self, image_batch, label_batch, is_train):
        if image_batch is None or label_batch is None:
            return {}
        drop_rate = 1.0
        if is_train:
            drop_rate = self._drop_rate
        images, labels = self.session.run([image_batch, label_batch])
        feed_dict = {
            self.input_x: images,
            self.input_y: labels,
            self.drop_rate_placehoder: drop_rate
        }
        return feed_dict

    def _training(self, image_batch, label_batch, writer=None):
        if image_batch is None or label_batch is None:
            return 0
        feed_dict = self._get_feed_dict(image_batch, label_batch, True)
        _, loss_val, summary, step = self.session.run([self.train_op, self.loss, self._merged_summary, self.global_step],
                                                      feed_dict=feed_dict)
        if writer is not None:
            writer.add_summary(summary, step)
        return loss_val, step

    def _get_accuracy(self, image_batch, label_batch, writer=None):
        if image_batch is None or label_batch is None:
            return 0
        feed_dict = self._get_feed_dict(image_batch, label_batch, False)
        acc, summary, step = self.session.run([self.accuracy, self._merged_summary, self.global_step],
                                              feed_dict=feed_dict)
        if writer is not None:
            writer.add_summary(summary, step)
        return acc

    def _evaluate_valid_accuracy(self, val_image_batch, val_label_batch, val_num_batches):
        if val_num_batches <= 0:
            return 0.0
        avg_acc = 0.0
        for i in range(val_num_batches):
            if i == 0:
                val_acc = self._get_accuracy(val_image_batch, val_label_batch, self._val_writer)
            else:
                val_acc = self._get_accuracy(val_image_batch, val_label_batch, None)
            avg_acc += val_acc
        avg_acc /= val_num_batches
        if avg_acc > self.get_best_val_acc():
            self._update_best_val_acc(avg_acc)
            self._save_model(model_name=self._BEST_MODEL_NAME)
        return avg_acc

    def _set_learning_rate(self, lr):
        if lr < self._min_learning_rate:
            lr = self._min_learning_rate
        elif lr > self._max_learning_rate:
            lr = self._max_learning_rate
        self.session.run(tf.assign(self.learning_rate, lr))

    def _protect_best_model(self):
        ckpt = tf.train.get_checkpoint_state(self._model_dir)
        all_model_checkpoint_paths = list()
        if ckpt:
            all_model_checkpoint_paths = ckpt.all_model_checkpoint_paths
        paths = list()
        best_model_path = list()
        for p in all_model_checkpoint_paths:
            if p.find(self._BEST_MODEL_NAME) < 0:
                paths.append(p)
            else:
                best_model_path.append(p)
        if best_model_path:
            paths.extend(best_model_path)
            self._saver.recover_last_checkpoints(paths)

    def get_best_val_acc(self):
        return self.session.run(self.best_val_acc)

    def _update_best_val_acc(self, avg_acc):
        self.session.run(tf.assign(self.best_val_acc, avg_acc))
