# Q Learning

- Q-learning is a value-based reinforcement learning algorithm. 
- It aims to learn the value of the optimal action-selection policy using the Q-function. 
- The Q-function is updated using the Bellman equation.

---

- Implementation of epsilon-greedy policy to balance exploration and exploitation.
- For each step in an episode, choose an action, perform the action, observe the reward and next state
- If the agent falls into a hole, update the reward to -1

---
 - After each episode, decay epsilon to reduce the exploration rate over time
 - Keep track of the total rewards per episode to analyze the learning progress
