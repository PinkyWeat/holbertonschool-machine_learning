#!/usr/bin/env python3
""" Bayesian Probability """
import numpy as np
intersection = __import__('1-intersection').intersection


def marginal(x, n, P, Pr):
    """ calculates the marginal probability of obtaining the data """

    if n < 1:
        raise ValueError("n must be a positive integer")
    if isinstance(x, int) and x <= 0:
        raise ValueError("n must be a positive integer")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) and P.shape[0] == 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) and Pr.shape[0] == P.shape[0]:
        raise TypeError("Pr must be a numpy.ndarray with the"
                        "same shape as P where {P} is the incorrect variable ")

    if not all(0 <= p <= 1 for p in P):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not all(0 <= pr <= 1 for pr in Pr):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    return np.sum(intersection(x, n, P, Pr))
