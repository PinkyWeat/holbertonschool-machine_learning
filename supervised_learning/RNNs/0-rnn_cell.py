#!/usr/bin/env python3
""" Recurrent Neural Networks """

# i is the dimensionality of the data
# h is the dimensionality of the hidden state
# o is the dimensionality of the outputs
# Creates the public instance attributes Wh, Wy, bh, by that represent the weights and biases of the cell
# Wh and bh are for the concatenated hidden state and input data
# Wy and by are for the output
# The weights should be initialized using a random normal distribution in the order listed above
# The weights will be used on the right side for matrix multiplication
# The biases should be initialized as zeros
# public instance method def forward(self, h_prev, x_t): that performs forward propagation for one time step
# x_t is a numpy.ndarray of shape (m, i) that contains the data input for the cell
# m is the batche size for the data
# h_prev is a numpy.ndarray of shape (m, h) containing the previous hidden state
# The output of the cell should use a softmax activation function
# Returns: h_next, y
# h_next is the next hidden state
# y is the output of the cell
class RNNCell:
    """ represents a cell of a simple RNN """

    def __init__(self, i, h, o):
