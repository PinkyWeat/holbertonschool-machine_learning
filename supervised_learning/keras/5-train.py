#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """ trains a model using mini-batch gradient descent """
    training_history = network.fit(x=data,
                                   y=labels,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   verbose=verbose,
                                   shuffle=shuffle,
                                   validation_data=validation_data)
    return training_history
