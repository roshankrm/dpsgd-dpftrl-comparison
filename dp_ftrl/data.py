"""Reading MNIST data."""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def get_mnist_data(batch_size):
    """Get MNIST dataset."""
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load training data
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    valid_size = 0.1
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    # Randomly shuffle indices
    np.random.shuffle(indices)

    # Split indices into training and validation
    train_idx, valid_idx = indices[split:], indices[:split]

    # Create Subset objects using these indices
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    valid_subset = torch.utils.data.Subset(train_dataset, valid_idx)

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    

    return train_loader, val_loader, test_loader, len(train_dataset), 10  # 10 classes in MNIST
