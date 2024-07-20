#!/usr/bin/env python3
""" RL - Policy Gradient """
import numpy as np


def policy(matrix, weight):
    """ computes the policy with a weight of a matrix """
    z = np.dot(matrix, weight)
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """ computes Monte Carlo poicy based on state & weight matrix """
    # compute the probs of taking each action given the current state
    probs = policy(state, weight)  # and weight matrix

    # randomly select an action based on the computed probs
    # the random choice selects an index=action according to given probs distr.
    action = np.random.choice(len(probs[0]), p=probs[0])

    # calculates the gradient
    # the probs of an action is, the likelihood that the action will be chosen
    dsoftmax = probs.copy()
    dsoftmax[0, action] -= 1
    # subtract 1 from the probs of the action that was actually taken
    # this is key; adjusting the gradient to focus on the diff between what
    # we got -chosen action- and what we wanted - the probabilities -

    # to see how changes in the weights, affect the selected action
    # we compute the dot product of the state and the adjusted probs
    gradient = np.dot(state.T, dsoftmax)
    # the dot product helps understand the influence of each state feature
    # in the decision
    # this gives us a matrix of gradients, which tells how to change the
    # weights to improve the policy.

    return action, gradient
