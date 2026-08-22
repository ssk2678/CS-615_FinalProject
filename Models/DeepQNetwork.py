from Linear import Linear
from ReLU import ReLU

class DQN:
    def __init__(self, state_dim, action_dim, lr=0.001):
        # Construct pipeline layers exactly matching your PyTorch configuration
        self.layer1 = Linear(state_dim, 64, lr)
        self.act1 = ReLU()
        self.layer2 = Linear(64, action_dim, lr)

    def forward(self, X):
        out = self.layer1.forward(X)
        out = self.act1.forward(out)
        out = self.layer2.forward(out)
        return out

    def backward(self, d_out):
        # Backpropagation passes gradients in exact reverse execution order
        grad = self.layer2.backward(d_out)
        grad = self.act1.backward(grad)
        grad = self.layer1.backward(grad)
        return grad

    def copy_weights_from(self, source_net):
        self.layer1.copy_weights_from(source_net.layer1)
        self.layer2.copy_weights_from(source_net.layer2)
