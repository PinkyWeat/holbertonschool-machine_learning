#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ saves them weights """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """ loads them weights """
    network.load_weights(filename)
