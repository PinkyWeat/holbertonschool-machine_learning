#!/usr/bin/env python3
"""Classification"""

import numpy as np


class Neuron():
    """Single neuron performs binary classification"""

    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def sigmoid(self, x):
        """non-linearity calc between 0 and 1"""
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        """calculates frwd propagation"""
        # res of combination X features and corresponding Weights + bias
        Z = np.dot(self.__W, X) + self.__b
        # non-linearity between 0 and 1
        self.__A = self.sigmoid(Z)
        return self.__A

    def cost(self, Y, A):
        """measures especificidad(?) of output based on labels"""
        # other def: calc cross-entropy loss
        # cost = -(1/m) * Σ [ Y * log(A) + (1 - Y) * log(1.0000001 - A) ]
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s prediction"""
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = X.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.dot(X, dZ.T)
        db = (1 / m) * np.sum(dZ)

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
