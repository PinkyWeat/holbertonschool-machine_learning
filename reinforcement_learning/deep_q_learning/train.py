#!/usr/bin/env python3
""" DQN - Training """
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

env = gym.make("Breakout-v0")

np.random.seed(123)
env.seed(123)

# creating neural network here
def build_model(state_shape, action_space):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + state_shape))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    return model

model = build_model(env.observation_space.shape, env.action_space.n)

# setting up DQN Agent to its policy & memory
def build_agent(model, action_space):
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=action_space, nb_steps_warmup=1000,
                   target_model_update=1e-2)
    return dqn

dqn = build_agent(model, env.action_space.n)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# training the agent
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# saving the model
dqn.save_weights('policy.h5', overwrite=True)
