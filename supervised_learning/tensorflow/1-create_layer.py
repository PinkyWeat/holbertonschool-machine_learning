#!/usr/bin/env python3
"""Task 1 - Create Layer"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """function creates a new layer"""

    # for the He et.al initialization of layer weights
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name="layer")
    return layer(prev)
