#!/usr/bin/env python3
""" Clustering """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ tests for the optimum number of clusters by variance """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if kmax is not None and (not isinstance(kmax, int) or kmax < 0):
        return None, None

    if not isinstance(kmin, int) or kmin <= 0 or kmax is not None \
            and kmin >= kmax:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []

    C, clss = kmeans(X, kmin, iterations)
    results.append((C, clss))
    d_vars.append(0.0)

    small_var = variance(X, C)
    kmin += 1

    if kmax is None:
        kmax = X.shape[0]
    while kmin <= kmax:
        C, clss = kmeans(X, kmin, iterations)
        d_vars.append(small_var - variance(X, C))
        results.append((C, clss))
        kmin += 1

    return results, d_vars
