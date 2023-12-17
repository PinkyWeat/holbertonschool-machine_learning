#!/usr/bin/env python3
""" Recurrent Neural Networks """
import numpy as np


class LSTMCell:
    """ represents a gated recurrent unit """

    def __init__(self, i, h, o):
        self.Wf = np.random.normal(size=(i+h, h))
        self.Wu = np.random.normal(size=(i+h, h))
        self.Wc = np.random.normal(size=(i+h, h))
        self.Wo = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """ performs forward propagation for one time step """
        def sigmoid(x):
            """Compute sigmoid activation function"""
            return 1 / (1 + np.exp(-x))

        h_x = np.concatenate((h_prev, x_t), axis=1)

        f_t = sigmoid(np.dot(h_x, self.Wf) + self.bf)

        u_t = sigmoid(np.dot(h_x, self.Wu) + self.bu)
        c_tilde = np.tanh(np.dot(h_x, self.Wc) + self.bc)

        c_next = f_t * c_prev + u_t * c_tilde

        output_t = sigmoid(np.dot(h_x, self.Wo) + self.bo)
        h_next = output_t * np.tanh(c_next)
        x = np.dot(h_next, self.Wy) + self.by
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        return h_next, c_next, y
