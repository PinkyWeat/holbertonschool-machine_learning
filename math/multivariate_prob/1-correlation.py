#!/usr/bin/env python3
""" Multivariate Probability """
import numpy as np


def correlation(C):
    """ calculates a correlation matrix """
    if not isinstance(C, numpy.ndarray) or C.ndim != 2:
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    variances = np.diag(C)
    standard_deviations = np.sqrt(variances)

    std_dev_product_matrix = np.outer(standard_deviations, standard_deviations)
    correlation_matrix = C / std_dev_product_matrix
    np.fill_diagonal(correlation_matrix, 1)

    return correlation_matrix
