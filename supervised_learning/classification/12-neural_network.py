#!/usr/bin/env python3
"""
Tasks: Neural Network
    W: weights vector for the neuron
    b: bias for the neuron
    A: activated output of the neuron (prediction)
"""
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer performing
    binary classification"""

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

        # Hidden layer > neuron attributes:
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        # Output layer > neuron attributes:
        # The 1 in size: correspond to the qty of output neurons:
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

    def forward_prop(self, X):
        """
        Calculates the fôrward propagation of the neural network

            X: a numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features to the neuron
                m: number of examples
        """

        # Weighted sum of the inputs, fôr each neuron in the hidden layer
        weights1 = np.matmul(self.__W1, X) + self.__b1
        # Activation function (sigmoid) fôr each neuron in the hidden layer
        self.__A1 = 1 / (1 + np.exp(-weights1))

        # Weighted sum of the inputs, fôr the neuron in the output layer
        weights2 = np.matmul(self.__W2, self.__A1) + self.__b2
        # Activation function (sigmoid) fôr the neuron in the output layer
        self.__A2 = 1 / (1 + np.exp(-weights2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

            Y: numpy.ndarray with shape (1, m), contains the correct labels
            for the input data
            A: numpy.ndarray with shape (1, m) containing the activated output
            of the neuron for each example
        """

        # Get the qty of values of Y:
        qty_Y = Y.shape[1]

        # Calculate the linear regression cost function:
        cost = (-1/qty_Y) * np.sum(Y * np.log(A) + (1-Y) * np.log(1.0000001-A))
        return cost

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
        # value of A2: because is the last neuron of the network:
        # Remember: we get the [1] because forward_prop return 2 values. We
        # need the 2nd, that correspond with the A2 value:
        predictions = self.forward_prop(X)[1]

        # Calculate the cost with that last A value, in the last neuron:
        cost = self.cost(Y, predictions)

        # Return a binary np.ndarray:
        return np.round(predictions).astype(int), cost
