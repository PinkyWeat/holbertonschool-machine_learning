#!/usr/bin/env python3
""" Dimensionality Reduction """
import numpy as np


def pca(X, var=0.95):
    """ performs PCA on a dataset """
    U, S, Vh = np.linalg.svd(X)

    V = Vh.T

    cumsum = np.cumsum(S)
    cumsum /= cumsum[-1]

    r = np.where(cumsum >= var)[0][0]

    return V[:, :r + 1]

