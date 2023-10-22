#!/usr/bin/env python3
""" DCA """
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ builds ResNet-50 archi as described in
    Deep Residual Learning for Image Recognition (2015) """
    X = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.HeNormal()

    # Initial Conv and MaxPool
    x = K.layers.Conv2D(64, 7,
                        strides=2,
                        padding="same",
                        kernel_initializer=initializer)(X)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # First set of layers
    x = projection_block(x, [64, 64, 256], s=1)
    for _ in range(2):
        x = identity_block(x, [64, 64, 256])

    # Second set of layers
    x = projection_block(x, [128, 128, 512])
    for _ in range(3):
        x = identity_block(x, [128, 128, 512])

    # Third set of layers
    x = projection_block(x, [256, 256, 1024])
    for _ in range(5):
        x = identity_block(x, [256, 256, 1024])

    # Fourth set of layers
    x = projection_block(x, [512, 512, 2048])
    for _ in range(2):
        x = identity_block(x, [512, 512, 2048])

    # Final layers
    x = K.layers.AveragePooling2D(7)(x)
    Y = K.layers.Dense(1000,
                       activation="softmax",
                       kernel_initializer=initializer)(x)

    model = K.Model(inputs=X, outputs=Y)

    return model
