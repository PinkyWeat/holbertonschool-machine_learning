#!/usr/bin/env python3
""" Convolutional Neural Network """
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """ builds a modified version of the LeNet-5 architecture using tf """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, (5, 5), padding='same', activation='relu', input_shape=x.shape[1:]),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model