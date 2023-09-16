#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """ trains a model using mini-batch gradient descent """
    training_history = network.fit(x=data,
                                   y=labels,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   verbose=verbose,
                                   shuffle=shuffle)
    return training_history
