#!/usr/bin/env python3
""" Optimization """


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ Updates variable using gradient descent w/ momentum op """
    momentum = beta1 * v + (1 - beta1) * grad
    # Update the variable using the momentum term
    var_new = var - alpha * momentum
    return var_new, momentum
