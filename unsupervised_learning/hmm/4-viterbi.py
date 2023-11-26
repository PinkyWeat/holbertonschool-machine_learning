#!/usr/bin/env python3
""" Hidden Markov Models """
import numpy as np
# Observation is a numpy.ndarray of shape (T,) that contains the index of
# the observation
# T is the number of observations
# Emission is a numpy.ndarray of shape (N, M) containing the emission
# probability of a specific observation given a hidden state
# Emission[i, j] is the probability of observing j given the hidden state i
# N is the number of hidden states
# M is the number of all possible observations
# Transition is a 2D numpy.ndarray of shape (N, N) containing the
# transition probabilities
# Transition[i, j] is the probability of transitioning from the hidden
# state i to j
# Initial a numpy.ndarray of shape (N, 1) containing the probability
# of starting in a particular hidden state
# Returns: path, P, or None, None on failure
# path is the a list of length T containing the most likely
# sequence of hidden states
# P is the probability of obtaining the path sequence


def viterbi(Observation, Emission, Transition, Initial):
    """ calculates the most likely sequence of hidden
    states for a hidden markov model """
    mu = np.zeros((len(Initial), len(Observation)))
    mu[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    path = np.zeros((len(Initial), len(Observation)), dtype=int)

    for t in range(1, len(Observation)):
        for j in range(len(Initial)):
            mu_j = mu[:, t - 1] * Transition[:, j] * \
                   Emission[j, Observation[t]]
            mu[j, t] = np.max(mu_j)
            path[j, t] = np.argmax(mu_j)

    P = np.max(mu[:, -1])

    best_path = [np.argmax(mu[:, -1])]
    for t in range(len(Observation) - 1, 0, -1):
        best_path.insert(0, path[best_path[0], t])

    return best_path, P
