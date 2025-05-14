"""Defines the neural network used for MNIST."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numbers import Number

def to_tuple(v, n):
    """Converts input to tuple."""
    if isinstance(v, tuple):
        return v
    elif isinstance(v, Number):
        return (v,) * n
    else:
        return tuple(v)

def objax_kaiming_normal(tensor, kernel_size, in_channels, out_channels, gain=1):
    """Objax's way of initializing using kaiming normal."""
    shape = (*to_tuple(kernel_size, 2), in_channels, out_channels)
    fan_in = np.prod(shape[:-1])
    kaiming_normal_gain = np.sqrt(1 / fan_in)
    std = gain * kaiming_normal_gain
    with torch.no_grad():
        return tensor.normal_(0, std)

def objax_initialize_conv(convs):
    """Objax's default initialization for conv2d."""
    for conv in convs:
        objax_kaiming_normal(conv.weight, conv.kernel_size, conv.in_channels, conv.out_channels)
        nn.init.zeros_(conv.bias)

def objax_initialize_linear(fcs):
    """Objax's default initialization for linear layer."""
    for fc in fcs:
        nn.init.xavier_normal_(fc.weight)
        nn.init.zeros_(fc.bias)

class SMALL_NN(nn.Module):
    def __init__(self, nclass=10):
        super(SMALL_NN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc1 = nn.Linear(32 * 16, 32)
        self.fc2 = nn.Linear(32, nclass)
        objax_initialize_conv([self.conv1, self.conv2])
        objax_initialize_linear([self.fc1, self.fc2])

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = torch.tanh(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = torch.tanh(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = torch.tanh(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

# Define LeNet architecture
class LeNet(nn.Module):
    def __init__(self, nclass=10):
        super(LeNet, self).__init__()
        # First convolutional layer: 1 input channel, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nclass)

    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16 * 5 * 5)
        # Fully connected layers with ReLU activation
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(nclass=10):
    """Get the MNIST model."""
    model = LeNet(nclass=nclass)
    
    # Initialize LeNet with objax initializations
    objax_initialize_conv([model.conv1, model.conv2])
    objax_initialize_linear([model.fc1, model.fc2, model.fc3])
    
    return model
