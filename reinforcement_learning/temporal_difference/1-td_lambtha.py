#!/usr/bin/env python3
""" Reinforcement Learning - TD Lambtha """
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """ Performs the TD(λ) algorithm to estimate the value function """
    for episode in range(episodes):
        state = env.reset(seed=0)[0] if isinstance(env.reset(seed=0), tuple) \
            else env.reset(seed=0)
        eligibility_traces = np.zeros_like(V)

        for step in range(max_steps):
            action = policy(state)
            result = env.step(action)
            next_state, reward, done = result[0], result[1], result[2]

            td_error = reward + gamma * V[next_state] - V[state]

            eligibility_traces[state] += 1  # update elegib* for current state

            V += alpha * td_error * eligibility_traces
            # update V func 4 all states

            # Decay eligibility traces
            eligibility_traces *= gamma * lambtha

            state = next_state

            if done:
                break

    return V
