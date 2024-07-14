#!/usr/bin/env python3
""" Reinforcement Learning - SARSA (l) """
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """ Performs SARSA(λ) on a given OpenAI gym environment """
    n_actions = env.action_space.n

    def epsilon_greedy_policy(state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(n_actions)
        else:
            return np.argmax(Q[state])

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        action = epsilon_greedy_policy(state, epsilon)
        E = np.zeros_like(Q)  # Initialize the eligibility trace

        for step in range(max_steps):
            next_state, reward, done, _ = env.step(action)[:4]
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            next_action = epsilon_greedy_policy(next_state, epsilon)

            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            E[state, action] += 1  # Increase the eligibility trace

            # Update Q values and eligibility traces
            Q += alpha * delta * E
            E *= gamma * lambtha

            state = next_state
            action = next_action

            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))

    return Q