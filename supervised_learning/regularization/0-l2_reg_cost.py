#!/usr/bin/env python3
""" Regularization """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m, l2_cost=None):
    """ calculates cost of a neural network with L2 regularization """
    l2_reg = 0

    for i in range(1, L + 1):
        l2_reg += np.linalg.norm(weights['W' + str(i)], 'fro')

        # Regularized cost
    reg_cost = cost + (lambtha / (2 * m)) * l2_reg

    return reg_cost
