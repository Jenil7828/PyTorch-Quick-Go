"""
Utility Functions for PyTorch Quick Go
=====================================

Common helper functions used across different modules:
- Model utilities
- Training utilities  
- Visualization helpers
- Data utilities
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os

def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }

def print_model_summary(model, input_shape=None):
    """Print a summary of the model architecture"""
    print("Model Architecture Summary")
    print("=" * 50)
    print(model)
    print("\nParameter Count:")
    
    params = count_parameters(model)
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    if params['frozen'] > 0:
        print(f"  Frozen parameters: {params['frozen']:,}")
    
    if input_shape:
        # Calculate model size estimation
        param_size = params['total'] * 4  # 4 bytes per parameter (float32)
        input_size = np.prod(input_shape) * 4  # 4 bytes per input value
        print(f"\nMemory Estimates (MB):")
        print(f"  Parameters: {param_size / 1e6:.2f}")
        print(f"  Input: {input_size / 1e6:.2f}")

def initialize_weights(model, init_type='xavier'):
    """Initialize model weights"""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(module.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif init_type == 'normal':
                nn.init.normal_(module.weight, 0, 0.02)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Timer:
    """Simple timer utility"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self):
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint"""
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {filename}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {filename}")
        return 0, float('inf')

def plot_training_curves(train_losses, val_losses=None, train_accs=None, val_accs=None, 
                        save_path='/tmp/training_curves.png'):
    """Plot training curves"""
    
    n_plots = 1 + (train_accs is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=3)
    if val_losses:
        axes[0].plot(val_losses, label='Val Loss', marker='s', markersize=3)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if train_accs is not None:
        axes[1].plot(train_accs, label='Train Acc', marker='o', markersize=3)
        if val_accs:
            axes[1].plot(val_accs, label='Val Acc', marker='s', markersize=3)
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")

def calculate_accuracy(outputs, targets):
    """Calculate accuracy from model outputs and targets"""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_learning_rate_scheduler(optimizer, scheduler_type='step', **kwargs):
    """Create a learning rate scheduler"""
    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=kwargs.get('step_size', 10), 
            gamma=kwargs.get('gamma', 0.5)
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=kwargs.get('T_max', 50)
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=kwargs.get('patience', 5),
            factor=kwargs.get('factor', 0.5)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def freeze_layers(model, freeze_embeddings=True, freeze_encoder=False):
    """Freeze specific layers in a model"""
    frozen_params = 0
    
    for name, param in model.named_parameters():
        if freeze_embeddings and 'embedding' in name.lower():
            param.requires_grad = False
            frozen_params += param.numel()
        elif freeze_encoder and 'encoder' in name.lower():
            param.requires_grad = False
            frozen_params += param.numel()
    
    print(f"Frozen {frozen_params:,} parameters")

def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def print_gpu_memory():
    """Print GPU memory usage if CUDA is available"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory - Allocated: {allocated:.1f} MB, Cached: {cached:.1f} MB")

def moving_average(data, window_size=5):
    """Calculate moving average of data"""
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        smoothed.append(np.mean(data[start_idx:i+1]))
    
    return smoothed

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def print_examples_header(title):
    """Print a formatted header for examples"""
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))

def example_usage():
    """Demonstrate usage of utility functions"""
    print_examples_header("Utility Functions Demo")
    
    # Create a simple model for demonstration
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    # Model summary
    print_model_summary(model, input_shape=(10,))
    
    # Initialize weights
    initialize_weights(model, 'xavier')
    
    # Timer example
    timer = Timer()
    timer.start()
    time.sleep(0.1)  # Simulate some work
    elapsed = timer.stop()
    print(f"\nTimer example: {elapsed:.3f} seconds")
    
    # Device selection
    device = get_device()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Model size
    size_mb = get_model_size_mb(model)
    print(f"Model size: {size_mb:.2f} MB")
    
    print("\nUtility functions demonstration completed!")

if __name__ == "__main__":
    example_usage()