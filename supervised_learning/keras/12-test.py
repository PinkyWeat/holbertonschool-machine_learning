#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ test them neural networks """
    evaluated = network.evaluate(x=data,
                                 y=labels,
                                 verbose=verbose)
    return evaluated