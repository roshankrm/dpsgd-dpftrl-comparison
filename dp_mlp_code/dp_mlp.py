import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directory for saving plots
PLOTS_DIR = "dp_mlp_plots"
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
    print(f"Created directory: {PLOTS_DIR}")

# Hyperparameters
BATCH_SIZE = 256  # Lot size in the paper's terminology
TEST_BATCH_SIZE = 512
N_EPOCHS = 20
LEARNING_RATE = 0.05 
MOMENTUM = 0.9
LOG_INTERVAL = 10
MAX_PHYSICAL_BATCH_SIZE = 512  # For memory management

# Privacy parameters
NOISE_MULTIPLIER = 1.0  # Sigma value
MAX_GRAD_NORM = 1.1  # Clipping threshold (C)
DELTA = 1e-5  # Target delta

# Define the MLP Classifier Network
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(MLP, self).__init__()
        # Create a list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten the input image
        x = x.view(x.size(0), -1)
        # Forward pass through the network
        return self.model(x)

def train(model, device, train_loader, optimizer, privacy_engine, epoch, disable_dp=False, delta=1e-5):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            losses.append(loss.item())
            
            # Update progress bar with current loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    # Calculate average loss and accuracy for the epoch
    avg_loss = sum(losses) / len(losses)
    accuracy = 100. * correct / total
    
    # Get privacy spent
    if not disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=delta)
        print(f"Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}% (ε = {epsilon:.2f}, δ = {delta})")
        return avg_loss, epsilon, accuracy
    else:
        print(f"Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, 0, accuracy

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy, all_preds, all_labels

def measure_inference_time(model, test_loader, device, num_runs=3):
    model.eval()
    
    # Warm-up run
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _ = model(images)
    
    # Measure inference time
    inference_times = []
    total_samples = len(test_loader.dataset)
    
    for run in range(num_runs):
        batch_times = []
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                
                # Measure batch inference time
                start_time = time.time()
                _ = model(images)
                end_time = time.time()
                
                batch_times.append(end_time - start_time)
        
        total_run_time = sum(batch_times)
        inference_times.append(total_run_time)
        print(f"Run {run+1}: {total_run_time:.4f} seconds")
    
    # Calculate statistics
    avg_inference_time = sum(inference_times) / len(inference_times)
    samples_per_second = total_samples / avg_inference_time
    ms_per_sample = (avg_inference_time * 1000) / total_samples
    
    print("\nInference Time Measurements:")
    print(f"  Test Dataset Size: {total_samples} samples")
    print(f"  Average Inference Time (over {num_runs} runs): {avg_inference_time:.4f}s")
    print(f"  Throughput: {samples_per_second:.2f} samples/second")
    print(f"  Latency: {ms_per_sample:.4f} ms/sample")
    
    return avg_inference_time


def main():
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
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
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(valid_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Define model parameters
    input_size = 28 * 28  # MNIST images are 28x28 pixels
    hidden_sizes = [512, 256, 128]  # Three hidden layers
    output_size = 10  # 10 classes (digits 0-9)
    dropout_rate = 0.2
    
    # Create model
    model = MLP(input_size, hidden_sizes, output_size, dropout_rate).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize PrivacyEngine
    privacy_engine = PrivacyEngine()
    
    # Make model, optimizer and data loader private
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM,
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epsilons = []
    
    start_time = time.time()
    
    for epoch in range(1, N_EPOCHS + 1):
        loss, eps, train_acc = train(model, device, train_loader, optimizer, privacy_engine, epoch, disable_dp=False, delta=DELTA)
        train_losses.append(loss)
        train_accs.append(train_acc)
        epsilons.append(eps)
        
        val_loss, val_acc, _, _ = test(model, device, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}: ε = {eps:.2f}, δ = {DELTA}")
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    
    # Final privacy accounting
    final_epsilon = privacy_engine.accountant.get_epsilon(delta=DELTA)
    print(f"Final privacy guarantee: (ε = {final_epsilon:.2f}, δ = {DELTA})")
    
    # Plot training and validation loss/accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    # Save to directory
    plt.savefig(os.path.join(PLOTS_DIR, 'sigma_1_dp_mlp_results.png'))
    plt.close()
    
    # Plot privacy budget growth
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epsilons) + 1), epsilons, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Privacy Budget (ε)')
    plt.title('Privacy Budget Growth')
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, 'sigma_1_dp_mlp_privacy_budget.png'))
    plt.close()
    
    # Final evaluation
    test_loss, test_acc, all_preds, all_labels = test(model, device, test_loader, criterion)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(PLOTS_DIR, 'sigma_1_dp_mlp_confusion_matrix.png'))
    plt.close()
    
        
    # Measure inference time on test set
    inference_time = measure_inference_time(model, test_loader, device)

if __name__ == "__main__":
    main()

