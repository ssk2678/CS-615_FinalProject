import numpy as np

class ReLU:
    def __init__(self):
        self.z_out = None  # Save output coordinates for masking

    def forward(self, X):
        self.z_out = X
        return np.maximum(0, X)

    def backward(self, d_out):
        # The derivative of ReLU is 1 if input > 0, else 0
        return d_out * (self.z_out > 0)