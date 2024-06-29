#!/usr/bin/env python3
""" Q Learning - Initialize Q Table """
import numpy as np


def q_init(env):
    """ initializes the Q-table """
    return np.zeros((env.observation_space.n, env.action_space.n))
