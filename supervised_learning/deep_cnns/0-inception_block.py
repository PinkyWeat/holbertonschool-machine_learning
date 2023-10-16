#!/usr/bin/env python3
""" DCA """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ builds an inception block as described in Going
    Deeper with Convolutions (2014) """
    conv1_0 = K.layers.Conv2D(filters[0], 1,
                              activation=K.activations.relu)(A_prev)

    conv1_1 = K.layers.Conv2D(filters[1], 1,
                              activation=K.activations.relu)(A_prev)
    conv3 = K.layers.Conv2D(filters[2], 3,
                            activation=K.activations.relu,
                            padding="same")(conv1_1)

    conv1_2 = K.layers.Conv2D(filters[3], 1,
                              activation=K.activations.relu)(A_prev)
    conv5 = K.layers.Conv2D(filters[4], 5,
                            activation=K.activations.relu,
                            padding="same")(conv1_2)

    max_pool = K.layers.MaxPool2D(3, 1, padding='same')(A_prev)
    conv1_3 = K.layers.Conv2D(filters[5], 1,
                              activation=K.activations.relu)(max_pool)

    return K.layers.Concatenate()([conv1_0, conv3, conv5, conv1_3])
