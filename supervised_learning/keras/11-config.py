#!/usr/bin/env python3
"""Tensorflow 2 & Keras"""
import tensorflow.keras as K


def save_config(network, filename):
    """ saves models configs to json """
    json_model = network.to_json()

    with open(filename, "w") as file:
        file.write(json_model)


def load_config(filename):
    """ loads them configs """
    with open(filename, "r") as file:
        json_model = file.read()

    return K.models.model_from_json(json_model)
