#!/usr/bin/env python3
""" Multivariate Probability """
import numpy as np

def mean_cov(X):
    """ calculates the mean and covariance of a data set """
    if len(X.shape) != 2 or not type(X) == np.array:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    # wip