#!/usr/bin/env python3
""" Optimization """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """  normalizes unactivated output of neural netwk
    using batch normalization """
    mu = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)

    # Normalize Z
    Z_norm = (Z - mu) / np.sqrt(var + epsilon)
    Z_out = gamma * Z_norm + beta

    return Z_out
