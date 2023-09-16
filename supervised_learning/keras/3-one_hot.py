#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ converts a label vector into a one-hot matrix """
    return K.utils.to_categorical(labels, num_classes=classes)
    # one liner ;)
