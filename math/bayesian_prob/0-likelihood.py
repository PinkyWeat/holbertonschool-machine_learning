#!/usr/bin/env python3
""" Bayesian Probability """
import numpy as np


def likelihood(x, n, P):
    """ calculates the likelihood of obtaining this data given various
        hypothetical probabilities of developing severe side effects"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x <= 0:
        raise ValueError("x must be an integer that is"
                         "greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    log_likelihoods = np.full(P.shape, -np.inf)

    mask = (P > 0) & (P < 1)

    log_likelihoods[mask] = ((np.log(np.arange(1, n + 1)).sum() -
                             (np.log(np.arange(1, x + 1)).sum() +
                              np.log(np.arange(1, n - x + 1)).sum())) +
                             x * np.log(P[mask]) + (n - x)
                             * np.log(1 - P[mask]))

    likelihoods = np.exp(log_likelihoods)

    return likelihoods
