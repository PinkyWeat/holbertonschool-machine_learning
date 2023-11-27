#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ performs the forward algorithm for a hidden markov model """
    alpha_t = np.zeros((len(Initial), len(Observation)))
    alpha_t[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, len(Observation)):
        for j in range(len(Initial)):
            alpha_t_j = np.sum(
                alpha_t[:, t - 1] * Transition[:, j]) * \
                        Emission[j, Observation[t]]
            alpha_t[j, t] = alpha_t_j

    P = np.sum(alpha_t[:, -1])

    return P, alpha_t


def backward(Observation, Emission, Transition, Initial):
    """ performs the backward algorithm for a hidden markov model """
    beta_t = np.zeros((len(Initial), len(Observation)))
    beta_t[:, -1] = 1

    for t in range(len(Observation) - 2, -1, -1):
        for j in range(len(Initial)):
            beta_t[j, t] = np.sum(
                beta_t[:, t + 1] * Transition[j, :] *
                Emission[:, Observation[t + 1]])

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta_t[:, 0])
    return P, beta_t


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
