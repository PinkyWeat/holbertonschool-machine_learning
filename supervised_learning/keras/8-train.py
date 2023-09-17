#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                early_stopping=False, patience=0,
                save_best=False, filepath=None,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                validation_data=None, verbose=True, shuffle=False):
    """ trains a model using mini-batch gradient descent
    also train the model using early stopping"""

    def learning_rate(epoch):
        """ calculates learning rate (only if validation_data exists) """
        return alpha / (1 + decay_rate * epoch)

    if validation_data:
        my_callbacks = []

        if early_stopping:
            my_callbacks.append(
                K.callbacks.EarlyStopping(monitor="val_loss",
                                          patience=patience))

        if learning_rate_decay:
            my_callbacks.append(
                K.callbacks.LearningRateScheduler(learning_rate, verbose=1))
    else:
        my_callbacks = None

    if save_best:
        if my_callbacks is None:
            my_callbacks = []

        my_callbacks.append(
            K.callbacks.ModelCheckpoint(filepath, save_best_only=True))

    training_history = network.fit(x=data,
                                   y=labels,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   verbose=verbose,
                                   shuffle=shuffle,
                                   validation_data=validation_data,
                                   callbacks=my_callbacks)
    return training_history
