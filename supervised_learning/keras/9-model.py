#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""
import tensorflow.keras as K


def save_model(network, filename):
    """ saves & loads model """
    network.save(filename)


def load_model(filename):
    """ loads model from file """
    return K.models.load_model(filename)
