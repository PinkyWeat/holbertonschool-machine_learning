#!/usr/bin/env python3
""" Bayesian Probability """
import numpy as np


def likelihood(x, n, P):
    """ calculates the likelihood of obtaining this data given various
        hypothetical probabilities of developing severe side effects"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not all(0 <= p <= 1 for p in P):
        raise ValueError("All values in P must be in the range [0, 1]")

    log_likelihoods = np.zeros(P.shape)

    for i, p in enumerate(P):
        log_likelihoods[i] = (np.math.factorial(n) / (np.math.factorial(x)
                                  * np.math.factorial(n - x))) * (p ** x) * \
              ((1 - p) ** (n - x))

    return log_likelihoods
