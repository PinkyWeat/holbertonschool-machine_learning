#!/usr/bin/env python3
"""Classification"""

import numpy as np
from matplotlib import pyplot as plt


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
        """Evaluates the neuron’s predictifon"""
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dz = A - Y
        dw = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m

        self.__W -= alpha * dw.T
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neuron"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        steps = []
        costs = []
        for i in range(iterations):
            # Frwd prop is performed to save the activated output A
            A = self.forward_prop(X)
            # Then, grandients of cost with respect of W and b are saved
            # W and b are updated using these gradients
            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

            if i % step == 0:
                cost = self.cost(Y, A)
                steps.append(i)
                costs.append(cost)
                if verbose:
                    print("Cost after {} iterations: {}".
                          format(i, cost))

            if graph:
                plt.title("Training Cost")
                plt.xlabel("iteration")
                plt.ylabel("cost")

                plt.plot(steps, costs)
                plt.show()

        # evaluation before anything
        return self.evaluate(X, Y)
