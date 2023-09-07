#!/usr/bin/env python3
""" Optimization """
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ Updates variable using gradient descent w/ momentum op """
    momentum = beta1 * v + (1 - beta1) * grad
    updated_value = var - (alpha - momentum)

    return momentum, updated_value
