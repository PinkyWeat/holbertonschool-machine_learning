#!/usr/bin/env python3
""" Convolutional Neural Network """
import tensorflow.compat.v1 as tf
tf.set_random_seed(0)


def lenet5(x, y):
    """ builds a modified version of the LeNet-5 architecture using tf """
    initializer = tf.variance_scaling_initializer(seed=0)

    conv1 = tf.layers.conv2d(x, 6, 5, padding='same',
                             use_bias=False,
                             kernel_initializer=initializer)
    bn1 = tf.layers.batch_normalization(conv1)
    act1 = tf.nn.relu(bn1)
    pool1 = tf.layers.max_pooling2d(act1, 2, 2)

    conv2 = tf.layers.conv2d(pool1, 16, 5,
                             use_bias=False,
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
    logits = tf.layers.dense(fc2, 10)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return tf.nn.softmax(logits), train_op, loss, accuracy
