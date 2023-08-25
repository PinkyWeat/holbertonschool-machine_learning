#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import pickle


class DeepNeuralNetwork():
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i, nodes in enumerate(layers):
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            in_dim = layers[i - 1] if i > 0 else nx
            std = np.sqrt(2 / in_dim)

            self.__weights[f"W{i+1}"] = np.random.randn(nodes, in_dim) * std
            self.__weights[f"b{i+1}"] = np.zeros((nodes, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    def forward_prop(self, X):
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            z = np.matmul(self.__weights[f"W{i}"], self.__cache[f"A{i-1}"]) \
                + self.__weights[f"b{i}"]
            self.__cache[f"A{i}"] = self.sigmoid(z)

        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        loss = -((Y * np.log(A)) + (1 - Y) * np.log(1.0000001 - A))
        return np.sum(loss) / m

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return (A > 0.5).astype(int), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        dz = cache[f"A{self.__L}"] - Y

        for i in range(self.__L, 0, -1):
            dw = np.matmul(dz, cache[f"A{i-1}"].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            dz = np.matmul(self.__weights[f"W{i}"].T, dz) \
                * (cache[f"A{i-1}"] * (1 - cache[f"A{i-1}"]))

            self.__weights[f"W{i}"] -= alpha * dw
            self.__weights[f"b{i}"] -= alpha * db

    def train(self, X, Y, iterations=1000, alpha=0.05,
              verbose=True, graph=True, step=100):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")

        costs = []
        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            if i % step == 0:
                cost = self.cost(Y, A)
                costs.append(cost)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

        if graph:
            plt.plot(range(0, iterations + 1, step), costs)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith
