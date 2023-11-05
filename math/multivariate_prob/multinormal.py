#!/usr/bin/env python3
""" Multivariate Probability """
import numpy as np
mean_cov = __import__("0-mean_cov").mean_cov


class MultiNormal():
    """ represents a Multivariate Normal distribution """
    def __init__(self, data):
        """ constructor stuff """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean, self.cov = mean_cov(data.T)

    def pdf(self, x):
        """ calculates the PDF at a data point """
        d = len(self.mean)

        if not isinstance(x, np.ndarray) or len(x.shape) != 2:
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        denom_const = (((2 * np.pi) ** (d / 2))
                       * (np.linalg.det(self.cov) ** 0.5))

        diff = x - self.mean
        exponent = -0.5 * np.linalg.solve(self.cov, diff).T.dot(diff)

        pdf = (1 / denom_const) * np.exp(exponent)
        return float(pdf.squeeze())
