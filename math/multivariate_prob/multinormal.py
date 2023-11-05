#!/usr/bin/env python3
""" Multivariate Probability """
import numpy as np
mean_cov = __import__("0-mean_cov").mean_cov


class MultiNormal():
    """ represents a Multivariate Normal distribution """
    def __init__(self, data):
        """ constructor stuff """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean, self.cov = mean_cov(data.T)
