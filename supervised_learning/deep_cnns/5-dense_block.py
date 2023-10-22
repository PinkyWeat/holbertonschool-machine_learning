#!/usr/bin/env python3
""" DCA """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ builds a dense block as described in
    Densely Connected Convolutional Networks """
    init = K.initializers.he_normal()

    for _ in range(layers):
        # Preserve the input tensor for concatenation
        X_shortcut = X

        # 1x1 Convolution
        X = K.layers.BatchNormalization()(X)
        X = K.layers.Activation("relu")(X)
        X = K.layers.Conv2D(4 * growth_rate, (1, 1),
                            padding="same",
                            kernel_initializer=init)(X)

        # 3x3 Convolution
        X = K.layers.BatchNormalization()(X)
        X = K.layers.Activation("relu")(X)
        X = K.layers.Conv2D(growth_rate, (3, 3),
                            padding="same",
                            kernel_initializer=init)(X)

        # Concatenatee feature maps from the previous
        # layer with the current one
        X = K.layers.Concatenate()([X_shortcut, X])

        nb_filters += growth_rate

    return X, nb_filters
