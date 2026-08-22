import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return (np.array([s[0] for s in samples]), 
                np.array([s[1] for s in samples]), 
                np.array([s[2] for s in samples]), 
                np.array([s[3] for s in samples]), 
                np.array([s[4] for s in samples]))
                
    def __len__(self):
        return len(self.buffer)
