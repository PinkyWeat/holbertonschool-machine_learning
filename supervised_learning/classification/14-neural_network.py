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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

            A1: output of the hidden layer
            A2: predicted output
            alpha: learning rate
        """

        # Get the quantity of training examples:
        qty_examples = Y.shape[1]

        # Last layer calculus
        # 1. Calculate the error:
        error_last = A2 - Y

        # 2. Calculate the gradient of the weights:
        weights_last = np.matmul(error_last, A1.T) / qty_examples

        # 3. Calculate the gradient of bias:
        bias_last = np.sum(error_last) / qty_examples

        # Hidden layer calculus
        # 1. Calculate the error:
        error_hidden = np.matmul(self.__W2.T, error_last) * (A1 * (1 - A1))

        # 2. Calculate the gradient of the weights:
        weights_hidden = np.matmul(error_hidden, X.T) / qty_examples

        # 3. Calculate the gradient of bias:
        bias_hidden = np.sum(error_hidden) / qty_examples

        # Update the parameters:
        self.__W2 -= alpha * weights_last
        self.__b2 -= alpha * bias_last
        self.__W1 -= alpha * weights_hidden
        self.__b1 -= alpha * bias_hidden

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network

            iterations: number of iterations to train over
            alpha: learning rate
        """

        # Parameter validations:
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        # Training loop f0r all neurons, all layers:
        for i in range(iterations):
            A1, A2 = self.forward_prop(X)

            # Update W and b in function to the product of the model train
            # implement backpropagation:
            self.gradient_descent(X, Y, A1, A2, alpha)

        # Final return: evaluation of the model performance:
        return self.evaluate(X, Y)
