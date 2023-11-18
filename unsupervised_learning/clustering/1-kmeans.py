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


def kmeans(X, k, iterations=1000):
    """ performs K-means on a dataset """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)

    for _ in range(iterations):
        C_prev = np.copy(C)

        # Vectorized distance calculation
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)

        # Updating centroids
        for i in range(k):
            cluster_points = X[cluster_assignments == i]
            if cluster_points.size > 0:
                C[i] = cluster_points.mean(axis=0)
            else:
                C[i] = initialize(X, 1)[0]  # Reinitialize if cluster is empty

        if np.all(C == C_prev):
            break

    return C, cluster_assignments
