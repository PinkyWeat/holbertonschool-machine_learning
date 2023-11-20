#!/usr/bin/env python3
""" Clustering """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ calculates the expectation step in the EM algorithm for a GMM """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    n, d = X.shape

    if (not isinstance(pi, np.ndarray)
            or len(pi.shape) != 1
            or not np.allclose(np.sum(pi), 1)):
        return None, None

    k = pi.shape[0]

    if not isinstance(m, np.ndarray) or len(m.shape) != 2 or \
            m.shape[0] != k or m.shape[1] != d:
        return None, None

    if not isinstance(S, np.ndarray) or len(S.shape) != 3 or \
            S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None

    g = np.zeros((k, n))
    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])
    m = np.sum(g, axis=0)

    likelihood = np.sum(np.log(m))
    g /= np.sum(g, axis=0)

    return g, likelihood
