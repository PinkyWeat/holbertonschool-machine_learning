#!/usr/bin/env python3
""" Regularization """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates the weights and biases of a neural network
    using gradient descent with L2 regularization """
    m = Y.shape[1]
    grads = {}

    # Loop from L-1 down to 0 to compute gradient descent
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y  # Gradient of softmax layer

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(i - 1)]

        # Update weights and bias
        weights["W" + str(i)] -= (alpha
                                  * (np.dot(dZ, A_prev.T) / m + (lambtha / m)
                                     * weights["W" + str(i)]))
        weights["b" + str(i)] -= alpha * np.sum(dZ, axis=1, keepdims=True) / m

        # gradient for next layer (stop before the input layer)
        if i > 1:
            dA = np.dot(weights["W" + str(i)].T, dZ)
            dZ = dA * (1 - cache["A" + str(i - 1)] ** 2)
            # Gradient for tanh activation
