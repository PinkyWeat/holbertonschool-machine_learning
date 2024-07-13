#!/usr/bin/env python3
""" Reinforcement Learning - Monte Carlo """
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """ performs Monte Carlo algorithm """
    for episode in range(episodes):
        state = env.reset()
        episode_data = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, reward))
            state = next_state
            if done:
                break

        G = 0
        for state, reward in reversed(episode_data):
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])

    return V
