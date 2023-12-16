#!/usr/bin/env python3
""" Recurrent Neural Networks """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """ performs forward propagation for a simple RNN """
    H = np.zeros((X.shape[0] + 1, h_0.shape[0], h_0.shape[1]))
    H[0] = h_0

    Y = np.zeros((X.shape[0], X.shape[1], rnn_cell.by.shape[1]))

    for i, x_t in enumerate(X):
        H[i + 1], Y[i] = rnn_cell.forward(H[i], x_t)

    return H, Y
