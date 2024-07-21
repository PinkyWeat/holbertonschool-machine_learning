#!/usr/bin/env python3
""" RL - Policy Gradient """
import numpy as np
from policy_gradient import policy, policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """ implements a full training """
    # random init for weights, will be adjusted during training
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)
    scores = []  # here we'll have sum of all rewards

    for episode in range(nb_episodes):
        state, _ = env.reset()
        state = state[None, :]
        ep_rewards = []
        done = False

        while not done:
            # getting action & gradient
            action, gradient = policy_gradient(state, weight)

            # taking action & observe result
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = next_state[None, :]  # adjustin shape compatibility

            # save reward
            ep_rewards.append(reward)

            # move to next state
            state = next_state

        # calculate sum of rewards = total score - for this episode
        score = sum(ep_rewards)
        scores.append(score)

        # prints current ep & score
        print(f"Episode {episode + 1}/{nb_episodes} - Score: {score}", end="\r", flush=True)

        for t, reward in enumerate(ep_rewards):
            Gt = sum([gamma ** i * r for i, r in enumerate(ep_rewards[t:])])  # Discounted future reward
            weight += alpha * gradient * Gt  # Update weights

    return scores
