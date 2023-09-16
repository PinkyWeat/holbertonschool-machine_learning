#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                early_stopping=False, patience=0,
                validation_data=None, verbose=True, shuffle=False):
    """ trains a model using mini-batch gradient descent
    also train the model using early stopping"""
    callbacks = [K.callbacks.EarlyStopping(monitor="val_loss",
                                           patience=patience)]\
        if validation_data and early_stopping else None

    training_history = network.fit(x=data,
                                   y=labels,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   verbose=verbose,
                                   shuffle=shuffle,
                                   validation_data=validation_data,
                                   callbacks=callbacks)
    return training_history
