#!/usr/bin/env python3
""" Clustering """
import numpy as np


def maximization(X, g):
    """ calculates the maximization step in the EM algorithm for a GMM """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    k, n1 = g.shape
    n, d = X.shape

    if n != n1 or np.abs(np.sum(g, axis=0) - 1).max() > 1e-10:
        return None, None, None

    S = np.zeros((k, d, d))

    sum_g = np.sum(g, axis=1)

    pi = sum_g / n
    m = np.dot(g, X) / sum_g[:, np.newaxis]

    for i in range(k):
        diff = X - m[i]
        weighted_diff = (g[i, :, np.newaxis] * diff).T
        S[i] = np.dot(weighted_diff, diff) / sum_g[i]

    return pi, m, S
