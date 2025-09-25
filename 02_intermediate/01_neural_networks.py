"""
PyTorch Intermediate: Neural Networks
====================================

This file covers building neural networks in PyTorch:
- nn.Module and building blocks
- Linear layers, activations, loss functions
- Forward pass implementation
- Model architecture design
- Initialization strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearModel(nn.Module):
    """Simple linear regression model"""
    
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

class MultiLayerPerceptron(nn.Module):
    """Multi-layer perceptron for classification"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super(MultiLayerPerceptron, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class CustomActivationModel(nn.Module):
    """Model demonstrating different activation functions"""
    
    def __init__(self, input_dim):
        super(CustomActivationModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        
        # Different activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.leaky_relu(self.layer2(x))
        x = self.tanh(self.layer3(x))
        x = self.output(x)
        return x

def basic_linear_model():
    """Demonstrate basic linear model"""
    print("=== Basic Linear Model ===")
    
    # Create model
    model = SimpleLinearModel(input_dim=5, output_dim=1)
    
    print("Model architecture:")
    print(model)
    
    # Model parameters
    print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters())}")
    
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    
    # Forward pass
    x = torch.randn(10, 5)  # Batch of 10 samples
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    return model

def mlp_classification():
    """Multi-layer perceptron for classification"""
    print("\n=== Multi-Layer Perceptron ===")
    
    # Create MLP
    mlp = MultiLayerPerceptron(
        input_dim=20,
        hidden_dims=[64, 32, 16],
        output_dim=3,  # 3 classes
        dropout_rate=0.3
    )
    
    print("MLP architecture:")
    print(mlp)
    
    # Count parameters
    total_params = sum(p.numel() for p in mlp.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Forward pass
    x = torch.randn(32, 20)  # Batch size 32
    logits = mlp(x)
    probabilities = F.softmax(logits, dim=1)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities sum (should be ~1): {probabilities.sum(dim=1)[:5]}")
    
    return mlp

def activation_functions_demo():
    """Demonstrate different activation functions"""
    print("\n=== Activation Functions ===")
    
    x = torch.linspace(-3, 3, 100)
    
    activations = {
        'ReLU': F.relu(x),
        'Leaky ReLU': F.leaky_relu(x, 0.1),
        'Tanh': torch.tanh(x),
        'Sigmoid': torch.sigmoid(x),
        'GELU': F.gelu(x),
        'Swish': x * torch.sigmoid(x)
    }
    
    print("Activation function ranges:")
    for name, values in activations.items():
        print(f"{name}: min={values.min():.3f}, max={values.max():.3f}")
    
    # Create model with custom activations
    model = CustomActivationModel(input_dim=10)
    
    # Test forward pass
    test_input = torch.randn(5, 10)
    output = model(test_input)
    print(f"\nCustom activation model output shape: {output.shape}")
    
    return activations

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.layer1(x))
        out = self.dropout(out)
        out = self.layer2(out)
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """Simple ResNet-style architecture"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3):
        super(ResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

def residual_networks():
    """Demonstrate residual networks"""
    print("\n=== Residual Networks ===")
    
    # Create ResNet
    resnet = ResNet(input_dim=50, hidden_dim=128, output_dim=10, num_blocks=4)
    
    print("ResNet architecture:")
    print(resnet)
    
    # Count parameters
    total_params = sum(p.numel() for p in resnet.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Forward pass
    x = torch.randn(16, 50)
    output = resnet(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    return resnet

def loss_functions_demo():
    """Demonstrate different loss functions"""
    print("\n=== Loss Functions ===")
    
    # Classification losses
    batch_size, num_classes = 10, 5
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Cross-entropy loss
    ce_loss = nn.CrossEntropyLoss()
    ce_value = ce_loss(logits, targets)
    print(f"Cross-entropy loss: {ce_value.item():.4f}")
    
    # NLL loss (requires log probabilities)
    log_probs = F.log_softmax(logits, dim=1)
    nll_loss = nn.NLLLoss()
    nll_value = nll_loss(log_probs, targets)
    print(f"NLL loss: {nll_value.item():.4f}")
    
    # Regression losses
    predictions = torch.randn(batch_size, 1)
    targets_reg = torch.randn(batch_size, 1)
    
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    huber_loss = nn.SmoothL1Loss()
    
    print(f"MSE loss: {mse_loss(predictions, targets_reg).item():.4f}")
    print(f"MAE loss: {mae_loss(predictions, targets_reg).item():.4f}")
    print(f"Huber loss: {huber_loss(predictions, targets_reg).item():.4f}")
    
    return ce_loss, mse_loss

def weight_initialization():
    """Demonstrate different weight initialization strategies"""
    print("\n=== Weight Initialization ===")
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # Xavier/Glorot initialization
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    # Create model and apply initialization
    model = MultiLayerPerceptron(input_dim=10, hidden_dims=[64, 32], output_dim=5)
    model.apply(init_weights)
    
    # Check weight statistics
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
    
    # Different initialization methods
    layer = nn.Linear(100, 50)
    
    # Kaiming/He initialization
    nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
    print(f"Kaiming init - mean: {layer.weight.mean().item():.4f}, std: {layer.weight.std().item():.4f}")
    
    # Normal initialization
    nn.init.normal_(layer.weight, mean=0, std=0.1)
    print(f"Normal init - mean: {layer.weight.mean().item():.4f}, std: {layer.weight.std().item():.4f}")
    
    return model

def model_saving_loading():
    """Demonstrate model saving and loading"""
    print("\n=== Model Saving and Loading ===")
    
    # Create and train a simple model
    model = SimpleLinearModel(input_dim=5, output_dim=1)
    
    # Save model state dict
    torch.save(model.state_dict(), '/tmp/model_state.pth')
    print("Model state dict saved")
    
    # Save entire model
    torch.save(model, '/tmp/model_complete.pth')
    print("Complete model saved")
    
    # Load model state dict
    new_model = SimpleLinearModel(input_dim=5, output_dim=1)
    new_model.load_state_dict(torch.load('/tmp/model_state.pth'))
    print("Model state dict loaded")
    
    # Verify models are identical
    x = torch.randn(1, 5)
    original_output = model(x)
    loaded_output = new_model(x)
    
    print(f"Outputs identical: {torch.allclose(original_output, loaded_output)}")
    
    return model, new_model

def training_example():
    """Complete training example"""
    print("\n=== Training Example ===")
    
    # Generate synthetic data
    torch.manual_seed(42)
    X = torch.randn(1000, 10)
    y = (X.sum(dim=1) > 0).long()  # Binary classification
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = MultiLayerPerceptron(input_dim=10, hidden_dims=[64, 32], output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if epoch % 2 == 0:
            print(f"Epoch [{epoch+1}/10], Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X)
        predictions = torch.argmax(test_outputs, dim=1)
        accuracy = (predictions == y).float().mean()
        print(f"Final accuracy: {accuracy.item():.4f}")
    
    return model, accuracy

def main():
    """Run all examples"""
    print("PyTorch Neural Networks Tutorial")
    print("=" * 40)
    
    basic_linear_model()
    mlp_classification()
    activation_functions_demo()
    residual_networks()
    loss_functions_demo()
    weight_initialization()
    model_saving_loading()
    training_example()
    
    print("\n" + "=" * 40)
    print("Tutorial completed successfully!")

if __name__ == "__main__":
    main()