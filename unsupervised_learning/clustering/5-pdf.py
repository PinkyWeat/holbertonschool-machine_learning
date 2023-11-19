#!/usr/bin/env python3
""" Clustering """
import numpy as np


def pdf(X, m, S):
    """ calculates probability density function of a Gaussian distribution """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    d = X.shape[1]

    if not isinstance(m, np.ndarray) or len(m.shape) != 1 or \
            m.shape[0] != d:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2 or \
            S.shape[0] != d or S.shape[1] != d:
        return None

    det_sqrt = np.linalg.det(S) ** 0.5
    inv_S = np.linalg.inv(S)

    diff = X - m
    exponent = -0.5 * np.einsum('ij,ji->i', diff @ inv_S, diff.T)

    P = (1 / ((2 * np.math.pi) ** (d / 2) * det_sqrt)) * np.exp(exponent)

    P = np.maximum(P, 1e-300)

    return P
