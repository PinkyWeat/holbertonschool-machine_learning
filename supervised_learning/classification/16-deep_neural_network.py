#!/usr/bin/env python3
"""
Tasks: Deep Neural Network
"""
import numpy as np


class DeepNeuralNetwork():
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Constructor class

            nx: number of input features
            layers: number of nodes in each layer of the network"""

        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Number of layers in the neural network:
        self.L = len(layers)
        # A dict to hold all intermediary values of the network:
        self.cache = {}
        # A dict to hold all weights and biases of the network:
        self.weights = {}

        for i in range(len(layers)):
            # Last validation:
            if type(layers[i]) != int or layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")

            columns = layers[i - 1] if i > 0 else nx

            # He et al. initialization:
            he_init = np.sqrt(2 / columns)

            # Get weights and bias randomly:
            Wi = np.random.randn(layers[i], columns) * he_init
            bi = np.zeros((layers[i], 1))

            # Set the values in weights dict:
            self.weights["W{}".format(i + 1)] = Wi
            self.weights["b{}".format(i + 1)] = bi
