#!/usr/bin/env python3
""" Q Learning - Load Environment """
import gym
from gym.envs.registration import register


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """ loads the pre-made FrozenLakeEnv environment from OpenAI’s gym """
    return gym.make('FrozenLake-v0', desc=desc,
                    map_name=map_name, is_slippery=is_slippery)
