#!/usr/bin/env python3
""" DCA """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ builds inception network from Going Deeper with Convolutions (2014) """
    init = K.initializers.HeNormal()
    relu = K.activations.relu
    Conv2D = K.layers.Conv2D
    MaxPooling2D = K.layers.MaxPooling2D
    AveragePooling2D = K.layers.AveragePooling2D
    Dropout = K.layers.Dropout
    Dense = K.layers.Dense

    X = K.Input(shape=(224, 224, 3))

    # Initial Conv and MaxPool layers
    x = Conv2D(64, 7, activation=relu,
               strides=2,
               kernel_initializer=init,
               padding="same")(X)
    x = MaxPooling2D(3, 2, padding="same")(x)

    # Conv layers before inception blocks
    x = Conv2D(64, 1, activation=relu,
               kernel_initializer=init,
               padding="same")(x)
    x = Conv2D(192, 3, activation=relu,
               kernel_initializer=init,
               padding="same")(x)
    x = MaxPooling2D(3, 2, padding="same")(x)

    # Inception blocks
    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])
    x = MaxPooling2D(3, 2, padding="same")(x)

    x = inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = MaxPooling2D(3, 2, padding="same")(x)

    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    # Average Pooling, Dropout and Dense (Softmax) layers
    x = AveragePooling2D(7, 1)(x)
    x = Dropout(0.4)(x)
    x = Dense(1000, activation=K.activations.softmax,
              kernel_initializer=init)(x)

    return K.Model(inputs=X, outputs=x)
