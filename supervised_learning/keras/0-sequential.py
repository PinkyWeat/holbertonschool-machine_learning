#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.python.keras import regularizers


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ building a neural network w/ Keras """
    model = Sequential()
    regularizer = regularizers.l2(lambtha)
    
    for i, node in enumerate(layers):
        if i == 0:
            model.add(Dense(node, activation=activations[i],
                            kernel_regularizer=regularizer, input_dim=nx))
        else:
            model.add(Dropout(1 - keep_prob))
            model.add(Dense(node, activation=activations[i],
                            kernel_regularizer=regularizer))
    return model
