#!/usr/bin/env python3
""" Clustering """
import numpy as np


def initialize(X, k):
    """ initializes cluster centroids for K-means """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    _, d = X.shape

    centroids = np.random.uniform(
        low=np.min(X, axis=0),
        high=np.max(X, axis=0),
        size=(k, d)
    )

    return centroids
