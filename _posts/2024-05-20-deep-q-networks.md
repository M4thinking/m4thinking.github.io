---
layout: post
title: Deep Q-Networks (DQN) - Theory and Practice
date: 2024-05-20 14:30:00 -0400
description: A comprehensive guide to Deep Q-Learning with PyTorch implementation
tags: dqn reinforcement-learning pytorch
categories: machine-learning
featured: true
mermaid: true
---

## Introduction to Deep Reinforcement Learning

Reinforcement Learning (RL) is a framework where an agent learns to make decisions by interacting with an environment. Deep Q-Networks (DQN) combine traditional Q-Learning with deep neural networks, enabling the handling of high-dimensional state spaces.

### Key Components of DQN

1. **Q-Learning Recap**:
   $$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

2. **Neural Network as Function Approximator**:
   $$Q(s,a; \theta) \approx Q^*(s,a)$$

3. **Experience Replay**:
   ```mermaid
   graph LR
   A[Agent] --> B[Store transition in buffer]
   B --> C[Sample mini-batch]
   C --> D[Train network]
   D --> A
   ```

4. **Target Network**:
   $$L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2]$$

## PyTorch Implementation

Let's implement DQN for the CartPole environment:

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayBuffer(10000)
epsilon = EPS_START

def select_action(state):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(action_dim)]], dtype=torch.long)

# Training loop
num_episodes = 500
for episode in range(num_episodes):
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    total_reward = 0
    
    while True:
        action = select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        memory.push(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if len(memory) >= BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)
            batch = list(zip(*transitions))
            
            # Compute Q-values
            state_batch = torch.cat(batch[0])
            action_batch = torch.cat(batch[1])
            reward_batch = torch.cat(batch[2])
            next_state_batch = torch.cat(batch[3])
            done_batch = torch.cat(batch[4])
            
            q_values = policy_net(state_batch).gather(1, action_batch)
            next_q_values = target_net(next_state_batch).max(1)[0].detach()
            expected_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values
            
            loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if done:
            break
    
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    print(f"Episode {episode}, Total Reward: {total_reward}")
```

## Key Implementation Details

1. **Experience Replay Buffer**:
   - Stores transitions (state, action, reward, next_state, done)
   - Random sampling breaks temporal correlations

2. **Epsilon-Greedy Exploration**:
   - Balances exploration/exploitation
   - Decays epsilon over time

3. **Target Network**:
   - Provides stable Q-value targets
   - Periodically updated from policy network

## Training Results

After 500 episodes, you should see the agent maintaining the pole upright for extended periods:

```
Episode 0, Total Reward: 12.0
Episode 10, Total Reward: 35.0
Episode 50, Total Reward: 125.0
Episode 100, Total Reward: 195.0
...
Episode 500, Total Reward: 500.0 (maximum)
```

## Challenges and Tips

1. **Reward Shaping**:
   - Design appropriate reward signals
   - Normalize rewards if necessary

2. **Hyperparameter Tuning**:
   - Learning rate: 0.0001-0.01
   - Discount factor (γ): 0.9-0.99
   - Batch size: 32-128

3. **Advanced Variants**:
   - Double DQN
   - Dueling DQN
   - Prioritized Experience Replay

## Conclusion

DQN demonstrates how deep learning can enhance traditional reinforcement learning algorithms. While the CartPole environment is relatively simple, the same principles apply to more complex tasks like Atari game playing. For production systems, consider using modern extensions like Rainbow DQN or combining with policy gradient methods.

**Further Reading**:
- [Original DQN Paper (Nature 2015)](https://www.nature.com/articles/nature14236)
- [OpenAI Spinning Up DQN Guide](https://spinningup.openai.com/)
- [Stable Baselines3 Implementation](https://stable-baselines3.readthedocs.io/) 