#!/usr/bin/env python3
""" Tasks: Deep Neural Network """
from matplotlib import pyplot as plt
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
        self.__L = len(layers)
        # A dict to hold all intermediary values of the network:
        self.__cache = {}
        # A dict to hold all weights and biases of the network:
        self.__weights = {}

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

    @property
    def L(self):
        """Private L attribute getter function"""
        return self.__L

    @property
    def cache(self):
        """Private cache attribute getter function"""
        return self.__cache

    @property
    def weights(self):
        """Private weights attribute getter function"""
        return self.__weights

    def sigmoid(self, X):
        """Sigmoid activation function"""
        return (1 / (1 + np.exp(-X)))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

            X: numpy.ndarray with shape (nx, m), contains the input data
                nx: number of input features to the neuron
                m: number of examples
        """

        cache, weights = self.__cache, self.__weights
        cache["A0"] = X
        for i in range(self.__L):
            Wi, bi = "W{}".format(i + 1), "b{}".format(i + 1)
            Ai, A_next = "A{}".format(i), "A{}".format(i + 1)

            # Weighted sum of the inputs, f0r each neuron:
            sum_weights = np.matmul(weights[Wi], cache[Ai]) + weights[bi]
            # Set each A value in cache using sigmoid activation function:
            cache[A_next] = self.sigmoid(sum_weights)

        return cache[A_next], cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

            Y: numpy.ndarray with shape (1, m), contains the correct labels
            for the input data
            A: numpy.ndarray with shape (1, m), contains the activated output
            of the neuron for each example
        """

        # Firstly, calculate the cross-entropy == loss function:
        loss = -((Y * np.log(A)) + (1 - Y) * (np.log(1.0000001 - A)))

        # Secondly, calculate the cost function == average of each loss result:
        return loss.mean()

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

            X: numpy.ndarray with shape (nx, m), contains the input data
                nx: number of input features to the neuron
                m: number of examples
            Y: numpy.ndarray with shape (1, m), contains the correct labels
            for the input data
        """

        # Forward propagation through all the neural network, and get the
        # value of A_next: because is the last neuron of the network:
        predictions = self.forward_prop(X)[0]

        # Calculate the cost with that last A value, in the last neuron:
        cost = self.cost(Y, predictions)

        # Return a binary np.ndarray:
        return np.round(predictions).astype(int), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

            alpha: learning rate
        """

        weights, layers = self.__weights, self.__L
        # Calculate the quantity of training examples:
        qty_examples = Y.shape[1]

        # Calculate the error in the LAST LAYER:
        error = cache["A" + str(layers)] - Y

        # Applies backpropagation, through the neural network from
        # the last layer:
        for i in range(layers, 0, -1):
            # Define the current W, b and A:
            Wi, bi = "W{}".format(i), "b{}".format(i)
            Ai = "A{}".format(i - 1)

            # Calculate the gradient of bias:
            bias_gradient = np.sum(error, axis=1, keepdims=True)/qty_examples

            # Calculate the gradient of the weights:
            weights_gradient = np.matmul(error, cache[Ai].T) / qty_examples

            # Update error in each layer:
            error = np.matmul(weights[Wi].T, error)*(cache[Ai]*(1-cache[Ai]))

            # Update the parameters:
            weights[Wi] -= alpha * weights_gradient
            weights[bi] -= alpha * bias_gradient

    def train(self, X, Y, iterations=1000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network

            iterations: number of iterations to train over
            verbose: boolean, defines whether to print information
            about the training
            graph: boolean, defines whether to graph information about
            the training once the training has completed
        """

        # Parameters validation:
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        elif iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")
        if type(step) != int:
            raise TypeError("step must be an integer")
        elif step < 1 or step <= iterations:
            raise ValueError("step must be positive and <= iterations")

        # Training loop f0r all neurons, all layers:
        costs, steps = [], []
        for i in range(iterations + 1):
            A_last, cache = self.forward_prop(X)

            # Update W and b in function to the product of the model train
            # implement backpropagation:
            self.gradient_descent(Y, cache, alpha)

            # Create the necessary arrays to create the chart:
            # Calculate costs only just each "step" steps, this avoid us
            # use plt.xticks in the chart construction:
            if i % step == 0:
                steps.append(i)
                costs.append(self.cost(Y, A_last))
                if verbose:
                    print("Cost after {} iterations: {}".
                          format(i, self.cost(Y, A_last)))

        if graph:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")

            plt.plot(steps, costs)
            plt.show()

        # Final return: evaluation of the model performance:
        return self.evaluate(X, Y)
