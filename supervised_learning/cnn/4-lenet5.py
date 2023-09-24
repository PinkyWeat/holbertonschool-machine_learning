#!/usr/bin/env python3
""" Convolutional Neural Network """
import tensorflow as tf


def lenet5(x, y):
    """ builds a modified version of the LeNet-5 architecture using tf """
    kernel_init = tf.contrib.layers.variance_scaling_initializer()

    layer_1 = tf.layers.Conv2D( # layer 1
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=kernel_init
    )
    layer1_result = layer_1(x)

    layer_max_pooling = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )
    layer_pooling_result = layer_max_pooling(layer1_result)

    layer_3 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=kernel_init
    )
    layer3_result = layer_3(layer_pooling_result)
    layer4_result = layer_max_pooling(layer3_result)

    flatten_result_4 = tf.layers.Flatten()(layer4_result)

    layer_fully_connected = tf.layers.Dense(
        120,
        activation=tf.nn.relu,
        kernel_initializer=kernel_init
    )
    layer_fully_result = layer_fully_connected(flatten_result_4)

    layer_fully_connected2 = tf.layers.Dense(
        84,
        activation=tf.nn.relu,
        kernel_initializer=kernel_init
    )
    layers_fully_resulting = layer_fully_connected2(layer_fully_result)

    layer_softmax_output = tf.layers.Dense(
        10,
        kernel_initializer=kernel_init
    )
    result_softmax_calc = layer_softmax_output(layers_fully_resulting)

    # outputs of outputs
    softmax_result = tf.nn.softmax(result_softmax_calc)
    loss = tf.losses.softmax_cross_entropy(y, logits=result_softmax_calc)
    training_op = tf.train.AdamOptimizer().minimize(loss)

    # accuracy calculated by output class label
    y_pred = tf.math.argmax(result_softmax_calc, axis=1)
    y_out = tf.math.argmax(y, axis=1)
    comparison = tf.math.equal(y_pred, y_out)
    accuracy = tf.reduce_mean(tf.cast(comparison, 'float'))

    return softmax_result, training_op, loss, accuracy
