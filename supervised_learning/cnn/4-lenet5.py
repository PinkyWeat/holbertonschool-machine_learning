#!/usr/bin/env python3
""" Convolutional Neural Network """
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """ builds a modified version of the LeNet-5 architecture using tf """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv1 = tf.layers.conv2d(x, 6, 5,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn1 = tf.layers.batch_normalization(conv1)
    act1 = tf.nn.relu(bn1)

    pool1 = tf.layers.max_pooling2d(act1, 2, 2)

    conv2 = tf.layers.conv2d(pool1, 16, 5,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn2 = tf.layers.batch_normalization(conv2)
    act2 = tf.nn.relu(bn2)

    pool2 = tf.layers.max_pooling2d(act2, 2, 2)
    flatten = tf.layers.flatten(pool2)

    fc1 = tf.layers.dense(flatten, 120,
                          activation=tf.nn.relu,
                          kernel_initializer=initializer)
    fc2 = tf.layers.dense(fc1, 84,
                          activation=tf.nn.relu,
                          kernel_initializer=initializer)

    logits = tf.layers.dense(fc2, 10,
                             activation=tf.math.softmax,
                             kernel_initializer=initializer)

    loss = tf.losses.softmax_cross_entropy(y, logits)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    pred_labels = tf.argmax(logits, axis=1)
    correct_pred = tf.equal(pred_labels, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return logits, train_op, loss, accuracy
