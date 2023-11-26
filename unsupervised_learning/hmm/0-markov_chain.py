#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np


def markov_chain(P, s, t=1):
    """ determines the probability of a markov chain being
    in a particular state after a specified number of iterations """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None

    if (not isinstance(s, np.ndarray) or len(s.shape) != 2
            or s.shape[1] != P.shape[0]):
        return None

    return s @ np.linalg.matrix_power(P, t)
