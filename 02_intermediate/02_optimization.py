"""
PyTorch Intermediate: Optimization and Training
==============================================

This file covers optimization techniques in PyTorch:
- Different optimizers (SGD, Adam, etc.)
- Learning rate scheduling
- Regularization techniques
- Training best practices
- Monitoring and debugging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class SimpleClassifier(nn.Module):
    """Simple classifier for optimization examples"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

def optimizer_comparison():
    """Compare different optimizers"""
    print("=== Optimizer Comparison ===")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Different optimizers to compare
    optimizers_config = {
        'SGD': {'class': optim.SGD, 'params': {'lr': 0.01, 'momentum': 0.9}},
        'Adam': {'class': optim.Adam, 'params': {'lr': 0.001}},
        'AdamW': {'class': optim.AdamW, 'params': {'lr': 0.001, 'weight_decay': 0.01}},
        'RMSprop': {'class': optim.RMSprop, 'params': {'lr': 0.001}},
    }
    
    results = {}
    
    for opt_name, opt_config in optimizers_config.items():
        print(f"\nTraining with {opt_name}...")
        
        # Create fresh model
        model = SimpleClassifier(input_dim=20, hidden_dim=64, output_dim=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = opt_config['class'](model.parameters(), **opt_config['params'])
        
        losses = []
        
        # Training loop
        model.train()
        for epoch in range(20):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
        
        results[opt_name] = losses
        print(f"Final loss: {losses[-1]:.4f}")
    
    return results

def learning_rate_scheduling():
    """Demonstrate learning rate scheduling"""
    print("\n=== Learning Rate Scheduling ===")
    
    # Create model and data
    model = SimpleClassifier(input_dim=10, hidden_dim=32, output_dim=2)
    X = torch.randn(500, 10)
    y = torch.randint(0, 2, (500,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Different schedulers
    schedulers = {
        'StepLR': optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5),
        'ExponentialLR': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
        'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20),
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    }
    
    for sched_name, scheduler in schedulers.items():
        print(f"\nTesting {sched_name}:")
        
        # Reset optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01
        
        lr_history = []
        
        for epoch in range(15):
            # Training step
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            
            # Update scheduler
            if sched_name == 'ReduceLROnPlateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: LR = {current_lr:.6f}, Loss = {avg_loss:.4f}")
    
    return lr_history

def regularization_techniques():
    """Demonstrate regularization techniques"""
    print("\n=== Regularization Techniques ===")
    
    # Generate data prone to overfitting
    X = torch.randn(200, 50)  # Small dataset, many features
    y = torch.randint(0, 2, (200,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    class RegularizedModel(nn.Module):
        def __init__(self, dropout_rate=0.5):
            super(RegularizedModel, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(50, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 2)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Train models with different regularization
    models = {
        'No Regularization': {'model': RegularizedModel(0.0), 'weight_decay': 0.0},
        'Dropout Only': {'model': RegularizedModel(0.3), 'weight_decay': 0.0},
        'L2 Regularization': {'model': RegularizedModel(0.0), 'weight_decay': 0.01},
        'Both': {'model': RegularizedModel(0.3), 'weight_decay': 0.01}
    }
    
    results = {}
    
    for name, config in models.items():
        print(f"\nTraining: {name}")
        
        model = config['model']
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=config['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        
        for epoch in range(30):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
        
        results[name] = losses[-1]
        print(f"Final loss: {losses[-1]:.4f}")
    
    return results

def gradient_clipping():
    """Demonstrate gradient clipping"""
    print("\n=== Gradient Clipping ===")
    
    # Create a model prone to exploding gradients
    class DeepModel(nn.Module):
        def __init__(self):
            super(DeepModel, self).__init__()
            layers = []
            for _ in range(10):  # Very deep network
                layers.extend([nn.Linear(50, 50), nn.ReLU()])
            layers.append(nn.Linear(50, 1))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    model = DeepModel()
    X = torch.randn(100, 50)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)  # High learning rate
    criterion = nn.MSELoss()
    
    print("Training with gradient clipping:")
    
    for epoch in range(10):
        total_loss = 0
        max_grad_norm = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Compute gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            max_grad_norm = max(max_grad_norm, grad_norm.item())
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Max grad norm = {max_grad_norm:.4f}")
    
    return model

def custom_optimizer():
    """Implement a custom optimizer"""
    print("\n=== Custom Optimizer ===")
    
    class CustomSGD(optim.Optimizer):
        """Custom SGD with momentum"""
        
        def __init__(self, params, lr=0.01, momentum=0.9):
            defaults = dict(lr=lr, momentum=momentum)
            super(CustomSGD, self).__init__(params, defaults)
        
        def step(self, closure=None):
            for group in self.param_groups:
                momentum = group['momentum']
                
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    d_p = p.grad.data
                    
                    if momentum != 0:
                        param_state = self.state[p]
                        if len(param_state) == 0:
                            param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                        d_p = buf
                    
                    p.data.add_(d_p, alpha=-group['lr'])
    
    # Test custom optimizer
    model = SimpleClassifier(input_dim=10, hidden_dim=32, output_dim=2)
    custom_opt = CustomSGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Compare with built-in SGD
    model_builtin = SimpleClassifier(input_dim=10, hidden_dim=32, output_dim=2)
    builtin_opt = optim.SGD(model_builtin.parameters(), lr=0.01, momentum=0.9)
    
    # Copy weights to ensure fair comparison
    model_builtin.load_state_dict(model.state_dict())
    
    # Single optimization step
    X = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    criterion = nn.CrossEntropyLoss()
    
    # Custom optimizer step
    custom_opt.zero_grad()
    loss_custom = criterion(model(X), y)
    loss_custom.backward()
    custom_opt.step()
    
    # Built-in optimizer step
    builtin_opt.zero_grad()
    loss_builtin = criterion(model_builtin(X), y)
    loss_builtin.backward()
    builtin_opt.step()
    
    print(f"Custom optimizer loss: {loss_custom.item():.6f}")
    print(f"Built-in optimizer loss: {loss_builtin.item():.6f}")
    
    return custom_opt

def training_monitoring():
    """Demonstrate training monitoring and debugging"""
    print("\n=== Training Monitoring ===")
    
    model = SimpleClassifier(input_dim=20, hidden_dim=64, output_dim=3)
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Split data
    train_size = int(0.8 * len(X_tensor))
    train_X, val_X = X_tensor[:train_size], X_tensor[train_size:]
    train_y, val_y = y_tensor[:train_size], y_tensor[train_size:]
    
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=32, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(20):
        # Training phase
        model.train()
        train_loss, train_correct = 0, 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        val_loss, val_correct = 0, 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if epoch % 5 == 0:
            print(f"Epoch [{epoch+1}/20]")
            print(f"  Train: Loss = {train_loss_avg:.4f}, Acc = {train_acc:.2f}%")
            print(f"  Val:   Loss = {val_loss_avg:.4f}, Acc = {val_acc:.2f}%")
    
    # Check for overfitting
    if val_losses[-1] > min(val_losses):
        print("\nWarning: Potential overfitting detected!")
    
    return train_losses, val_losses, train_accs, val_accs

def early_stopping():
    """Implement early stopping"""
    print("\n=== Early Stopping ===")
    
    class EarlyStopping:
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
    
    # Train with early stopping
    model = SimpleClassifier(input_dim=15, hidden_dim=32, output_dim=2)
    X = torch.randn(400, 15)
    y = torch.randint(0, 2, (400,))
    
    train_X, val_X = X[:300], X[300:]
    train_y, val_y = y[:300], y[300:]
    
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=32)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Higher LR for faster overfitting
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=5)
    
    for epoch in range(50):  # Maximum epochs
        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        val_loss /= len(val_loader)
        
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")
    
    return model, early_stopping

def main():
    """Run all examples"""
    print("PyTorch Optimization Tutorial")
    print("=" * 40)
    
    optimizer_comparison()
    learning_rate_scheduling()
    regularization_techniques()
    gradient_clipping()
    custom_optimizer()
    training_monitoring()
    early_stopping()
    
    print("\n" + "=" * 40)
    print("Tutorial completed successfully!")

if __name__ == "__main__":
    main()