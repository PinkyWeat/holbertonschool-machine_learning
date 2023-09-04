#!/usr/bin/env python3
""" Optimization """
import numpy as np


def shuffle_data(X, Y):
    """ shuffles data points in 2 matrix """
    perm = np.random.permutation(len(X))
    return X[perm], Y[perm]
