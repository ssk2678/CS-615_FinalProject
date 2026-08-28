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
        # Optimized 2-episode training loop constants
        self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay = 0.99, 1.0, 0.05, 0.998

    def select_action(self, state):
        if random.random() < self.epsilon: 
            return random.randint(0, self.action_dim - 1)
        
        # Reshape into a 2D batch array, then flatten the output to guarantee a 1D scalar index
        q_values = self.policy_net.forward(state.reshape(1, -1)).flatten()
        return np.argmax(q_values)

    def train_step(self, batch_size):
        if len(self.memory) < batch_size: 
            return None
            
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # 1. Compute Bellman Target Expected Q-Values using frozen Target Network
        target_q_values = self.target_net.forward(next_states)
        max_next_q = np.max(target_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # 2. Forward pass through active Policy Network
        current_q_values = self.policy_net.forward(states)
        selected_q_values = current_q_values[np.arange(batch_size), actions]
        loss = np.mean((selected_q_values - targets) ** 2)
        
        # 3. Initialize loss gradient at the final layer output
        d_out = np.zeros_like(current_q_values)
        for i in range(batch_size):
            # Batch_size division to perfectly balance with Linear.py's Adam updates
            d_out[i, actions[i]] = (current_q_values[i, actions[i]] - targets[i]) / batch_size
            
        # 4. Fire backward pass sequence (Gradients auto-cascade and parameters self-update)
        self.policy_net.backward(d_out)
        
        if self.epsilon > self.epsilon_min: 
            self.epsilon *= self.epsilon_decay
        return float(loss)
