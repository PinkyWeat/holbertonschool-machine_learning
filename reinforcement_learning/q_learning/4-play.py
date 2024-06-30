#!/usr/bin/env python3
""" Q Learning - Play """
import numpy as np


def play(env, Q, max_steps=100):
    """ has the trained agent play an episode """
    ep_reward = 0
    state = env.reset()  # reset at beginning

    for step in range(max_steps):
        print(env.render(mode='ansi'))  # display's current state of board

        # exploit == select the action with max Q-value
        action = np.argmax(Q[state])

        # take action, observe the outcome
        next_state, reward, done, _ = env.step(action)

        ep_reward += reward  # sum of rewards

        state = next_state  # update state to current

        if done:  # given episode's done, exit
            break

    print(env.render(mode='ansi'))  # display's final state of board
    return ep_reward
