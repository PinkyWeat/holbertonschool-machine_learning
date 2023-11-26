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
