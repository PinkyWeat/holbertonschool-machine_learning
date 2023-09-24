#!/usr/bin/env python3
""" Convolutional Neural Network """
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """ builds a modified version of the LeNet-5 architecture using tf """
    initializer = tf.variance_scaling_initializer()

    conv1 = tf.layers.conv2d(x, 6, 5, padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    conv2 = tf.layers.conv2d(pool1, 16, 5,
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    flatten = tf.layers.flatten(pool2)
    fc1 = tf.layers.dense(flatten, 120,
                          activation=tf.nn.relu,
                          kernel_initializer=initializer)
    fc2 = tf.layers.dense(fc1, 84,
                          activation=tf.nn.relu,
                          kernel_initializer=initializer)
    logits = tf.layers.dense(fc2, 10)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return tf.nn.softmax(logits), train_op, loss, accuracy
