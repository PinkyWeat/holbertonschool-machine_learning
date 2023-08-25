#!/usr/bin/env python3
""" Tasks: One-Hot Encode """
import numpy as np


def one_hot_encode(Y, classes):
    """ Converts a numeric label vector into a one-hot matrix """
    try:
        one_hot_matrix = np.zeros((classes, Y.shape[0]))
        one_hot_matrix[Y, np.arange(Y.shape[0])] = 1
        return one_hot_matrix
    except Exception as e:
        return None
