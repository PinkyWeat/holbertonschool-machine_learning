#!/usr/bin/env python3
""" RL - Policy Gradient """
import numpy as np


def policy(matrix, weight):
    """ computes the policy with a weight of a matrix """
    z = np.dot(matrix, weight)
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# The function calculates the "policy" for an agent
# The policy, tells the agent the probability of taking each action
# given a certain state.
# Using a weight matrix to help compute this.
