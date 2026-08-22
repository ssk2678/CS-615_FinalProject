import numpy as np

class Linear:
    def __init__(self, in_features, out_features, lr=0.001):
        self.lr = lr
        # He (Kaiming) Parameter Initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        
        # Tracking inputs needed to compute local weight gradients during backprop
        self.X_in = None
        
        # Adam Optimizer Tracking States
        self.beta1, self.beta2, self.eps, self.t = 0.9, 0.999, 1e-8, 0
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)

    def forward(self, X):
        self.X_in = X  # Save inputs for the backward pass chain rule
        return np.dot(X, self.W) + self.b

    def backward(self, d_out):
        """
        Input: d_out -> Gradient of loss with respect to this layer's output (batch_size, out_features)
        Output: d_X  -> Gradient of loss passed backward to the previous layer (batch_size, in_features)
        """
        batch_size = d_out.shape[0]
        
        # Compute local parameter gradients
        dW = np.dot(self.X_in.T, d_out)
        db = np.sum(d_out, axis=0, keepdims=True)
        
        # Compute structural input gradient to pass further back up the network chain
        d_X = np.dot(d_out, self.W.T)
        
        # Execute isolated Adam Parameter Adjustments
        self.t += 1
        
        # Update Weights (W)
        self.mW = self.beta1 * self.mW + (1 - self.beta1) * dW
        self.vW = self.beta2 * self.vW + (1 - self.beta2) * (dW ** 2)
        mW_hat = self.mW / (1 - self.beta1 ** self.t)
        vW_hat = self.vW / (1 - self.beta2 ** self.t)
        self.W -= (self.lr * mW_hat) / (np.sqrt(vW_hat) + self.eps)
        
        # Update Biases (b)
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * db
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * (db ** 2)
        mb_hat = self.mb / (1 - self.beta1 ** self.t)
        vb_hat = self.vb / (1 - self.beta2 ** self.t)
        self.b -= (self.lr * mb_hat) / (np.sqrt(vb_hat) + self.eps)
        
        return d_X

    def copy_weights_from(self, source_layer):
        self.W = np.copy(source_layer.W)
        self.b = np.copy(source_layer.b)
