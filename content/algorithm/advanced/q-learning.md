---
title: Q-Learning
---
```python
import gym  
import numpy as np  
  
# Create a Q-table to represent state-action values  
states = 10  
actions = 2  
q_table = np.zeros((states, actions))  
  
# Define Q-learning parameters  
learning_rate = 0.1  
discount_factor = 0.9  
exploration_prob = 0.2  
  
# Implement Q-learning algorithm  
for episode in range(1000):  
state = env.reset()  
done = False  
  
while not done:  
# Choose action using epsilon-greedy policy  
action = epsilon_greedy_policy(q_table, state, exploration_prob)  
  
# Take the chosen action  
next_state, reward, done, _ = env.step(action)  
  
# Update Q-value using the Q-learning formula  
update_q_value(q_table, state, action, reward, next_state, learning_rate, discount_factor)  
  
# Move to the next state  
state = next_state
```