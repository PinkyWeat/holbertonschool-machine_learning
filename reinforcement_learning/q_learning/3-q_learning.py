#!/usr/bin/env python3
""" Q Learning - Q-learning """
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """ performs Q-learning """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()  # at beginning, reset each time
        episode_reward = 0

        for step in range(max_steps):
            if np.random.rand() < epsilon:
                # explore == select a random action
                action = env.action_space.sample()
            else:
                # exploit == select the action with max Q-value
                action = np.argmax(Q[state])

            try:  # try to handle different versions of the environment
                next_state, reward, done, _ = env.step(action)
            except ValueError:
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated

            if done and reward == 0:
                reward = -1  # if it falls in a hole, let it know, and learn

            # q learning update
            Q[state, action] += alpha * (reward + gamma
                                         * np.max(Q[next_state])
                                         - Q[state, action])

            state = next_state
            episode_reward += reward

            if done:
                break

        # Epsilon's dek
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))

        # tracking of total rewards
        total_rewards.append(episode_reward)

    return Q, total_rewards
