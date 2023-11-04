#!/usr/bin/env python3
""" Multivariate Probability """
import numpy as np


def mean_cov(X):
    """ calculates the mean and covariance of a data set """
    if len(X.shape) != 2 or not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    cov = X_centered.T @ X_centered / (X.shape[0] - 1)
    return mean, cov
