#!/usr/bin/env python3
""" Convolutional Neural Network """
import tensorflow.keras as K


def lenet5(X):
    """ builds modified version of Lenet5 archi w/ keras """
    model = K.Sequential()
    model._set_inputs(X)

    model.add(K.layers.Conv2D(6, 5, padding='same',
                              activation=K.activations.relu,
                              kernel_initializer=K.initializers.HeNormal))
    model.add(K.layers.MaxPool2D(2, 2))

    model.add(K.layers.Conv2D(16, 5, padding='valid',
                              activation=K.activations.relu,
                              kernel_initializer=K.initializers.HeNormal))
    model.add(K.layers.MaxPool2D(2, 2))

    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(120, K.activations.relu,
                             kernel_initializer=K.initializers.HeNormal))
    model.add(K.layers.Dense(84, K.activations.relu,
                             kernel_initializer=K.initializers.HeNormal))
    model.add(K.layers.Dense(10, K.activations.softmax,
                             kernel_initializer=K.initializers.HeNormal))

    adam = K.optimizers.Adam()
    model.compile(loss="categorical_crossentropy", optimizer=adam,
                  metrics=['accuracy'])

    return model
