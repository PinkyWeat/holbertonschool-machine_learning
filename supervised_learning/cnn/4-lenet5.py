#!/usr/bin/env python3
""" Convolutional Neural Network """
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """ builds a modified version of LeNet-5 architecture with tensorflow """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv_1 = tf.layers.conv2d(x, 6, 5, padding="same", activation=tf.nn.relu,
                              kernel_initializer=initializer)
    max_pool_1 = tf.layers.max_pooling2d(conv_1, 2, 2)
    conv_2 = tf.layers.conv2d(max_pool_1, 16, 5, padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=initializer)
    max_pool_2 = tf.layers.max_pooling2d(conv_2, 2, 2)
    flatten = tf.layers.flatten(max_pool_2)
    full_connected_1 = tf.layers.dense(flatten, 120, activation=tf.nn.relu,
                                       kernel_initializer=initializer)
    full_connected_2 = tf.layers.dense(full_connected_1, 84,
                                       activation=tf.nn.relu,
                                       kernel_initializer=initializer)
    y_pred = tf.layers.dense(full_connected_2, 10,
                             kernel_initializer=initializer)
    output = tf.nn.softmax(y_pred)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    pred_labels = tf.argmax(output, axis=1)
    correct_pred = tf.equal(pred_labels, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return output, train, loss, accuracy
