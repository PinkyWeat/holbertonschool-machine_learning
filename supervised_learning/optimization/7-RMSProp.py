#!/usr/bin/env python3
""" Optimization """
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ updates a variable using RMSProp optimization algorithm """

    # moving avg of sqaured grads
    s = beta2 * s + (1 - beta2) * np.power(grad, 2)
    var_updated = var - alpha * grad / (np.sqrt(s) + epsilon)

    return var_updated, s
