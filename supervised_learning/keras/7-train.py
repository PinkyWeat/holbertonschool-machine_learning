#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                validation_data=None, verbose=True, shuffle=False):
    """ trains a model using mini-batch gradient descent
    also train the model using early stopping"""

    def learning_rate(epoch):
        """ calculates learning rate (only if validation_data exists) """
        return alpha / (1 + decay_rate * epoch)

    if validation_data:
        callbacks = []

        if early_stopping:
            callbacks.append(
                K.callbacks.EarlyStopping(monitor="val_loss",
                                          patience=patience))

        if learning_rate_decay:
            callbacks.append(
                K.callbacks.LearningRateScheduler(learning_rate, verbose=1))
    else:
        callbacks = None

    training_history = network.fit(x=data,
                                   y=labels,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   verbose=verbose,
                                   shuffle=shuffle,
                                   validation_data=validation_data,
                                   callbacks=callbacks)
    return training_history
