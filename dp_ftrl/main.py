# -----------------------------------------------------------------------------
# This code uses the privacy computation and optimizer from:
#
#   google-research/DP-FTRL  
#   https://github.com/google-research/DP-FTRL

# -----------------------------------------------------------------------------

"""Main script for DP-FTRL on MNIST."""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import time
import matplotlib.pyplot as plt

from opacus import PrivacyEngine

from optimizers import FTRLOptimizer
from ftrl_noise import NoiseTreeAggregator
from nn import get_model
from data import get_mnist_data
from privacy import compute_epsilon_tree

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def plot_metrics(epochs_range, train_values, valid_values, metric_name, save_path):
    """
    Plot and save train and validation metrics.
    
    Args:
        epochs_range: List of epoch numbers
        train_values: List of training metric values
        valid_values: List of validation metric values
        metric_name: Name of the metric (e.g., 'Loss', 'Accuracy')
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_values, 'bo-', label=f'Training {metric_name}')
    plt.plot(epochs_range, valid_values, 'ro-', label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_privacy_budget(epochs_range, epsilon_values, save_path):
    """
    Plot and save privacy budget (epsilon) over epochs.
    
    Args:
        epochs_range: List of epoch numbers
        epsilon_values: List of epsilon values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, epsilon_values, 'go-', label='Privacy Budget (ε)')
    plt.title('Privacy Budget over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Epsilon (ε)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    # Setup random seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.run - 1)
    np.random.seed(args.run - 1)

    # Data
    train_loader, valid_loader, test_loader, ntrain, nclass = get_mnist_data(args.batch_size)
    print('Training set size:', ntrain)

    # Hyperparameters for training
    epochs = args.epochs
    batch = args.batch_size
    num_batches = ntrain // batch
    noise_multiplier = args.noise_multiplier if args.dp_ftrl else -1
    clip = args.l2_norm_clip if args.dp_ftrl else -1
    lr = args.learning_rate

    # Create log directory
    log_dir = os.path.join(args.dir, f"mnist_batch{batch}_noise{noise_multiplier}_clip{clip}_lr{lr}_run{args.run}")
    os.makedirs(log_dir, exist_ok=True)
    print('Model dir:', log_dir)

    # Function for evaluating the model
    def test(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        print(f'Test loss: {test_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')
        return accuracy, test_loss

    # Get model
    model = get_model(nclass=nclass).to(device)

    # Set the (DP-)FTRL optimizer
    base_optimizer = FTRLOptimizer(model.parameters(), momentum=args.momentum,
                                  record_last_noise=False)
    
    # Store reference to original optimizer
    original_optimizer = base_optimizer

    if args.dp_ftrl:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=base_optimizer,
            data_loader=train_loader,
            noise_multiplier=0.0,  # We'll add noise through the noise generator
            max_grad_norm=clip,
        )
    else:
        optimizer = base_optimizer

    shapes = [p.shape for p in model.parameters()]

    def get_cumm_noise():
        if not args.dp_ftrl or noise_multiplier == 0:
            return lambda: [torch.zeros(1).to(device)] * len(shapes)  # just return zeros
        return NoiseTreeAggregator(noise_multiplier * clip / batch, shapes, device)

    cumm_noise = get_cumm_noise()

    # Function to conduct training for each epoch
    def train_loop(model, optimizer, original_optimizer, cumm_noise, epoch):
        model.train()
        criterion = nn.CrossEntropyLoss(reduction='mean')
        losses = []
        correct = 0
        total = 0

        loop = trange(len(train_loader), desc=f'Epoch {1 + epoch}/{epochs}', leave=False)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Set parameters in the original optimizer
            original_optimizer.set_params(lr, cumm_noise())

            # Use standard step call that works with Opacus wrapper
            optimizer.step()

            losses.append(loss.item())

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            loop.update(1)

        avg_loss = np.mean(losses)
        accuracy = correct / total
        print(f'Epoch {epoch + 1} Loss {avg_loss:.4f} Accuracy {100.0 * accuracy:.2f}%')
        return avg_loss, accuracy

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
        print(f" Test Dataset Size: {total_samples} samples")
        print(f" Average Inference Time (over {num_runs} runs): {avg_inference_time:.4f}s")
        print(f" Throughput: {samples_per_second:.2f} samples/second")
        print(f" Latency: {ms_per_sample:.4f} ms/sample")

        return avg_inference_time

    # Training loop
    all_train_losses = []
    all_train_accs = []
    all_valid_losses = []
    all_valid_accuracies = []
    
    # List to track epsilon values per epoch
    all_epsilon_values = []
    
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_loop(model, optimizer, original_optimizer, cumm_noise, epoch)
        all_train_losses.append(train_loss)
        all_train_accs.append(train_acc)
        
        valid_acc, valid_loss = test(model, valid_loader)
        all_valid_accuracies.append(valid_acc)
        all_valid_losses.append(valid_loss)
        
        # Calculate epsilon for the current epoch
        if args.dp_ftrl and noise_multiplier > 0:
            epsilon = compute_epsilon_tree(
                num_batches=num_batches,
                epochs_between_restarts=[epoch + 1],  # Current number of epochs
                noise=noise_multiplier,
                delta=1e-5,
                verbose=False
            )
            all_epsilon_values.append(epsilon)
            print(f'Current privacy guarantee: (ε={epsilon:.2f}, δ=1e-5)-DP')
        
        print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%, Validation Accuracy: {valid_acc * 100:.2f}%')
        
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    
    # Plot and save the metrics
    epochs_range = list(range(1, epochs + 1))
    
    # Plot and save training and validation accuracy
    plot_metrics(
        epochs_range, 
        [acc * 100 for acc in all_train_accs],  # Convert to percentage
        [acc * 100 for acc in all_valid_accuracies],  # Convert to percentage
        'Accuracy (%)', 
        os.path.join(log_dir, 'accuracy_plot.png')
    )
    
    # Plot and save training and validation loss
    plot_metrics(
        epochs_range, 
        all_train_losses, 
        all_valid_losses, 
        'Loss', 
        os.path.join(log_dir, 'loss_plot.png')
    )
    
    # Plot and save privacy budget if DP-FTRL is used
    if args.dp_ftrl and noise_multiplier > 0 and all_epsilon_values:
        plot_privacy_budget(
            epochs_range, 
            all_epsilon_values, 
            os.path.join(log_dir, 'privacy_budget_plot.png')
        )
    
    # Print final privacy guarantee
    if args.dp_ftrl and noise_multiplier > 0:
        epsilon = compute_epsilon_tree(
            num_batches=num_batches,
            epochs_between_restarts=[epochs],
            noise=noise_multiplier,
            delta=1e-5,
        )
        print(f'Final privacy guarantee: (ε={epsilon:.2f}, δ=1e-5)-DP')

    # Final evaluation on test set
    print("\nFinal Evaluation on Test Set:")
    test_accuracy, test_loss = test(model, test_loader)
    print(f"Test Accuracy: {100.0 * test_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Measure inference time on test set
    inference_time = measure_inference_time(model, test_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DP-FTRL for MNIST')
    parser.add_argument('--dp_ftrl', action='store_true', help='Train with DP-FTRL')
    parser.add_argument('--noise_multiplier', type=float, default=1.0, help='Noise multiplier')
    parser.add_argument('--l2_norm_clip', type=float, default=1.1, help='Clipping norm')
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum for DP-FTRL')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=250, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--run', type=int, default=1, help='Run ID (for random seed)')
    parser.add_argument('--dir', type=str, default='./results', help='Directory to save results')

    args = parser.parse_args()
    main(args)
