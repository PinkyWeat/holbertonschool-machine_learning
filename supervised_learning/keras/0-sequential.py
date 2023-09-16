#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""
import tensorflow as tf
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ building a neural network w/ Keras """
    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)
    
    for i, node in enumerate(layers):
        if i == 0:
            model.add(K.layers.Dense(node, activation=activations[i],
                            kernel_regularizer=regularizer, input_dim=nx))
        else:
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(node, activation=activations[i],
                            kernel_regularizer=regularizer))
    return model
