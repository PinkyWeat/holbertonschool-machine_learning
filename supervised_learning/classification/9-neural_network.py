#!/usr/bin/env python3
""" Neural Network """

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden
    layer performing binary classification"""

    def __init__(self, nx, nodes):
        """
        Constructor class

            nx: number of input features
            nodes: number of nodes in the hidden layer
        """

        # Parameters validation:
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif type(nodes) != int:
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer attributes:
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Output layer attributes:
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Private W1 attribute getter function"""
        return self.__W1

    @property
    def b1(self):
        """Private b1 attribute getter function"""
        return self.__b1

    @property
    def A1(self):
        """Private A1 attribute getter function"""
        return self.__A1

    @property
    def W2(self):
        """Private W2 attribute getter function"""
        return self.__W2

    @property
    def b2(self):
        """Private b2 attribute getter function"""
        return self.__b2

    @property
    def A2(self):
        """Private A2 attribute getter function"""
        return self.__A2
