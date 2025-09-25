"""
PyTorch Basics: Data Loading and Datasets
=========================================

This file covers PyTorch's data loading capabilities:
- Dataset and DataLoader classes
- Custom datasets
- Data transformations
- Batch processing
- Common data loading patterns
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression

class CustomDataset(Dataset):
    """Custom dataset class implementation"""
    
    def __init__(self, features, targets, transform=None):
        """
        Args:
            features: Input features
            targets: Target values
            transform: Optional transform to be applied on a sample
        """
        self.features = features
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {
            'features': self.features[idx],
            'targets': self.targets[idx]
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class NormalizeFeatures:
    """Transform to normalize features"""
    
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        features = sample['features']
        
        if self.mean is None:
            self.mean = features.mean()
        if self.std is None:
            self.std = features.std()
        
        sample['features'] = (features - self.mean) / self.std
        return sample

def basic_tensor_dataset():
    """Demonstrate TensorDataset usage"""
    print("=== Basic TensorDataset ===")
    
    # Create sample data
    X = torch.randn(100, 4)  # 100 samples, 4 features
    y = torch.randint(0, 2, (100,))  # Binary classification
    
    # Create TensorDataset
    dataset = TensorDataset(X, y)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"First sample: {dataset[0]}")
    print(f"Feature shape: {dataset[0][0].shape}")
    print(f"Target shape: {dataset[0][1].shape}")
    
    return dataset

def basic_dataloader():
    """Demonstrate DataLoader usage"""
    print("\n=== Basic DataLoader ===")
    
    # Create sample data
    X = torch.randn(1000, 10)
    y = torch.randint(0, 3, (1000,))
    dataset = TensorDataset(X, y)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    print(f"Number of batches: {len(dataloader)}")
    
    # Iterate through batches
    for i, (batch_X, batch_y) in enumerate(dataloader):
        print(f"Batch {i}: X shape = {batch_X.shape}, y shape = {batch_y.shape}")
        if i >= 2:  # Show only first 3 batches
            break
    
    return dataloader

def custom_dataset_example():
    """Demonstrate custom dataset implementation"""
    print("\n=== Custom Dataset Example ===")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create custom dataset with transforms
    transform = NormalizeFeatures()
    custom_dataset = CustomDataset(X_tensor, y_tensor, transform=transform)
    
    print(f"Custom dataset size: {len(custom_dataset)}")
    
    # Get a sample
    sample = custom_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Features shape: {sample['features'].shape}")
    print(f"Target: {sample['targets']}")
    
    # Create DataLoader from custom dataset
    custom_dataloader = DataLoader(custom_dataset, batch_size=16, shuffle=True)
    
    for batch in custom_dataloader:
        print(f"Batch features shape: {batch['features'].shape}")
        print(f"Batch targets shape: {batch['targets'].shape}")
        break
    
    return custom_dataset

def train_validation_split():
    """Demonstrate train/validation split"""
    print("\n=== Train/Validation Split ===")
    
    # Create dataset
    X = torch.randn(1000, 8)
    y = torch.randint(0, 4, (1000,))
    dataset = TensorDataset(X, y)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Total dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader

class RegressionDataset(Dataset):
    """Custom dataset for regression problems"""
    
    def __init__(self, n_samples=1000, n_features=10, noise=0.1):
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=42
        )
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # Add dimension for target
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def regression_dataset_example():
    """Example with regression dataset"""
    print("\n=== Regression Dataset Example ===")
    
    # Create regression dataset
    reg_dataset = RegressionDataset(n_samples=500, n_features=5)
    
    print(f"Regression dataset size: {len(reg_dataset)}")
    
    # Check sample
    X_sample, y_sample = reg_dataset[0]
    print(f"Feature sample shape: {X_sample.shape}")
    print(f"Target sample shape: {y_sample.shape}")
    print(f"Target sample value: {y_sample.item():.4f}")
    
    # Create DataLoader
    reg_loader = DataLoader(reg_dataset, batch_size=64, shuffle=True)
    
    # Show batch statistics
    for X_batch, y_batch in reg_loader:
        print(f"Batch X shape: {X_batch.shape}")
        print(f"Batch y shape: {y_batch.shape}")
        print(f"Target mean: {y_batch.mean().item():.4f}")
        print(f"Target std: {y_batch.std().item():.4f}")
        break
    
    return reg_dataset

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data"""
    
    def __init__(self, data, sequence_length=10):
        """
        Args:
            data: 1D tensor of time series data
            sequence_length: Length of input sequences
        """
        self.data = data
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        # Input sequence
        x = self.data[idx:idx + self.sequence_length]
        # Target (next value)
        y = self.data[idx + self.sequence_length]
        return x, y

def time_series_example():
    """Example with time series dataset"""
    print("\n=== Time Series Dataset Example ===")
    
    # Generate synthetic time series
    t = torch.linspace(0, 4*np.pi, 200)
    time_series = torch.sin(t) + 0.1 * torch.randn(200)
    
    # Create time series dataset
    seq_length = 20
    ts_dataset = TimeSeriesDataset(time_series, sequence_length=seq_length)
    
    print(f"Time series length: {len(time_series)}")
    print(f"Dataset size: {len(ts_dataset)}")
    
    # Check sample
    x_seq, y_target = ts_dataset[0]
    print(f"Input sequence shape: {x_seq.shape}")
    print(f"Target shape: {y_target.shape}")
    
    # Create DataLoader
    ts_loader = DataLoader(ts_dataset, batch_size=16, shuffle=False)
    
    for x_batch, y_batch in ts_loader:
        print(f"Batch input shape: {x_batch.shape}")
        print(f"Batch target shape: {y_batch.shape}")
        break
    
    return ts_dataset

def data_loading_best_practices():
    """Demonstrate data loading best practices"""
    print("\n=== Data Loading Best Practices ===")
    
    # Create larger dataset
    X = torch.randn(10000, 20)
    y = torch.randint(0, 5, (10000,))
    dataset = TensorDataset(X, y)
    
    # Best practices for DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=128,  # Reasonable batch size
        shuffle=True,    # Shuffle for training
        num_workers=0,   # Parallel data loading (set to 0 for compatibility)
        pin_memory=True if torch.cuda.is_available() else False,  # Speed up GPU transfer
        drop_last=True   # Drop incomplete batch
    )
    
    print(f"DataLoader configuration:")
    print(f"- Batch size: {dataloader.batch_size}")
    print(f"- Shuffle: {dataloader.shuffle}")
    print(f"- Drop last: {dataloader.drop_last}")
    print(f"- Pin memory: {dataloader.pin_memory}")
    
    # Iterate through a few batches
    batch_sizes = []
    for i, (batch_x, batch_y) in enumerate(dataloader):
        batch_sizes.append(batch_x.size(0))
        if i >= 10:
            break
    
    print(f"First 10 batch sizes: {batch_sizes}")
    
    return dataloader

def collate_function_example():
    """Demonstrate custom collate function"""
    print("\n=== Custom Collate Function ===")
    
    def custom_collate(batch):
        """Custom collate function for variable-length sequences"""
        # Separate inputs and targets
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in inputs)
        padded_inputs = []
        
        for seq in inputs:
            if len(seq) < max_len:
                padding = torch.zeros(max_len - len(seq))
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq
            padded_inputs.append(padded_seq)
        
        return torch.stack(padded_inputs), torch.tensor(targets)
    
    # Create variable-length sequences
    sequences = []
    for i in range(50):
        length = torch.randint(5, 15, (1,)).item()
        seq = torch.randn(length)
        target = torch.randint(0, 2, (1,)).item()
        sequences.append((seq, target))
    
    # Use custom collate function
    custom_loader = DataLoader(
        sequences,
        batch_size=8,
        shuffle=True,
        collate_fn=custom_collate
    )
    
    for batch_x, batch_y in custom_loader:
        print(f"Padded batch shape: {batch_x.shape}")
        print(f"Target batch shape: {batch_y.shape}")
        break
    
    return custom_loader

def main():
    """Run all examples"""
    print("PyTorch Data Loading Tutorial")
    print("=" * 40)
    
    basic_tensor_dataset()
    basic_dataloader()
    custom_dataset_example()
    train_validation_split()
    regression_dataset_example()
    time_series_example()
    data_loading_best_practices()
    collate_function_example()
    
    print("\n" + "=" * 40)
    print("Tutorial completed successfully!")

if __name__ == "__main__":
    main()