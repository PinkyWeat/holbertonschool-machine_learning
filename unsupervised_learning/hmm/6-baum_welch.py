#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np
forward = __import__('3-forward').forward
backward = __import__('5-backward').backward


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """ performs the Baum-Welch algorithm for a hidden markov model """
    try:
        T, = Observations.shape
        M, N = Emission.shape

        for i in range(iterations):
            Pf, alpha = forward(Observations, Emission, Transition, Initial)
            Pb, beta = backward(Observations, Emission, Transition, Initial)

            xi = np.zeros((M, M, T - 1))

            for t in range(T - 1):
                denominator = np.dot(np.dot(alpha[:, t].T, Transition) *
                                     Emission[:, Observations[t + 1]].T,
                                     beta[:, t + 1])
                for j in range(M):
                    numerator = alpha[j, t] * Transition[j, :] * \
                                Emission[:, Observations[t + 1]].T * \
                                beta[:, t + 1].T
                    xi[j, :, t] = numerator / denominator

            gamma = np.sum(xi, axis=1)
            Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
            gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((
                -1, 1))))

            for k in range(N):
                Emission[:, k] = np.sum(gamma[:, Observations == k], axis=1)
            Emission = np.divide(Emission, np.sum(gamma, axis=1).reshape((
                -1, 1)))

        return Transition, Emission

    except Exception as exception:
        return None, None
