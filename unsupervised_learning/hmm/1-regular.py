#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np


def regular(P):
    """ determines the steady state probabilities of a regular markov chain """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2 or \
            np.count_nonzero(P == 0):
        return None

    eigen_values, eigen_vectors = np.linalg.eig(P)

    eigen_one_idxs = [round(eigen.real) for eigen in eigen_values].index(1.0)

    s = (eigen_vectors[:, eigen_one_idxs]
         / np.sum(eigen_vectors[:, eigen_one_idxs], axis=0))
    prev_s = 0

    while not (np.array_equal(s, prev_s)):
        prev_s = s
        s = prev_s @ P

    return s.reshape(1, len(P))
