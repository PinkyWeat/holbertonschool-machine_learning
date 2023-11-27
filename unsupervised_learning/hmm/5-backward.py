#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np


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
