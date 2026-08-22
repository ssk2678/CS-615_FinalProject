import random
import numpy as np
from DeepQNetwork import DQN
from ReplayBuffer import ReplayBuffer

class Agent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.copy_weights_from(self.policy_net)
        
        self.memory = ReplayBuffer(10000)
        self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay = 0.99, 1.0, 0.05, 0.995

    def select_action(self, state):
        if random.random() < self.epsilon: 
            return random.randint(0, self.action_dim - 1)
        q_values = self.policy_net.forward(state.reshape(1, -1))
        return np.argmax(q_values)

    def train_step(self, batch_size):
        if len(self.memory) < batch_size: 
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # 1. Compute Bellman Target Expected Q-Values using frozen Target Network
        target_q_values = self.target_net.forward(next_states)
        max_next_q = np.max(target_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # 2. Forward pass through active Policy Network
        current_q_values = self.policy_net.forward(states)
        
        # 3. Initialize loss gradient at the final layer output
        d_out = np.zeros_like(current_q_values)
        for i in range(batch_size):
            d_out[i, actions[i]] = (current_q_values[i, actions[i]] - targets[i]) / batch_size
            
        # 4. Fire backward pass sequence (Gradients auto-cascade and parameters self-update)
        self.policy_net.backward(d_out)
        
        if self.epsilon > self.epsilon_min: 
            self.epsilon *= self.epsilon_decay
