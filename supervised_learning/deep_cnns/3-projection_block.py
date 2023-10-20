#!/usr/bin/env python3
""" DCA """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ reates a tensorflow layer that includes L2 regularization """
    initializer = K.initializers.he_normal()
    F11, F3, F12 = filters

    # First 1x1 Convolution, BatchNorm, and ReLU activation
    conv1_0 = K.layers.Conv2D(F11, 1,
                              strides=s,
                              kernel_initializer=initializer)(A_prev)
    bn1_0 = K.layers.BatchNormalization()(conv1_0)
    relu1_0 = K.layers.Activation(K.activations.relu)(bn1_0)

    # 3x3 Convolution, BatchNorm, and ReLU activation
    conv3 = K.layers.Conv2D(F3, 3,
                            padding='same',
                            kernel_initializer=initializer)(relu1_0)
    bn3 = K.layers.BatchNormalization()(conv3)
    relu3 = K.layers.Activation(K.activations.relu)(bn3)

    # Second 1x1 Convolution and BatchNorm
    conv1_1 = K.layers.Conv2D(F12, 1, kernel_initializer=initializer)(relu3)
    bn1_1 = K.layers.BatchNormalization()(conv1_1)

    # Shortcut 1x1 Convolution and BatchNorm
    shortcut = K.layers.Conv2D(F12, 1,
                               strides=s,
                               kernel_initializer=initializer)(A_prev)
    bn_shortcut = K.layers.BatchNormalization()(shortcut)

    # Add main path and shortcut
    add = K.layers.Add()([bn1_1, bn_shortcut])

    return K.layers.Activation(K.activations.relu)(add)
