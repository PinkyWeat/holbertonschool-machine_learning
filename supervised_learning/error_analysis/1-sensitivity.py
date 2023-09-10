#!/usr/bin/env python3
"""
Task 1. Sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """ calculates sensitivity for each class in confusion matrixm"""

    id_matrix = np.identity(confusion.shape[1])
    P = np.sum(confusion, axis=1)
    return TP / P
