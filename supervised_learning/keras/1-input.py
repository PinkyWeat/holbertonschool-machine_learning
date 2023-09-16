#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ builds a neural network with Keras """
    regularizer = K.regularizers.l2(lambtha)
    input_layer = K.Input(shape=(nx,))
    more_layers = input_layer

    for i, node in enumerate(layers):
        if i != 0:
            more_layers = K.layers.Dropout(1 - keep_prob)(more_layers)
        more_layers = K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=regularizer
                                     )(more_layers)

    model = K.Model(inputs=input_layer, outputs=more_layers)
    return model
