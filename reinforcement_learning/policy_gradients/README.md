# Policy Gradients

### Why Calculate the Gradient?

> In reinforcement learning, we need to update our weights to make better decisions over time.

To do this, we need to know how changing the weights, will change our action probabilities. This is where the gradient comes in. The gradient tells us the direction and rate at which the probabilities change when we adjust the weights.

Basically, Doctor Strange thangs'

---

### Interesting philosophical discussion here...

`Found this:`

##### _Why Discount Future Rewards?_

`Uncertainty:` In many real-world scenarios, the future is uncertain. Immediate rewards are more certain than future rewards.

`Mathematical Stability:` Discounting helps to ensure that the sum of rewards (which can be infinite in some cases) remains finite.

`Preference for Immediate Feedback:` In many tasks, getting immediate feedback helps the agent learn more effectively.

_imo: could be interesting debate stitching it with Social Media._