"""
PyTorch Computer Vision: CNN Basics
===================================

This file covers fundamental CNN concepts and implementations:
- Convolutional layers and operations
- Pooling layers
- Basic CNN architectures
- Image preprocessing and data augmentation
- Transfer learning basics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class BasicCNN(nn.Module):
    """Basic CNN for image classification"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(BasicCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # Note: This assumes 32x32 input images (like CIFAR-10)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.dropout1(x)
        
        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.dropout1(x)
        
        # Third conv block
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class LeNet5(nn.Module):
    """LeNet-5 architecture"""
    
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        # Conv layer 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Conv layer 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class CustomCNNBlock(nn.Module):
    """Custom CNN block with batch normalization"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CustomCNNBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ModernCNN(nn.Module):
    """Modern CNN with batch normalization and residual connections"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(ModernCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = CustomCNNBlock(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Conv blocks
        self.conv2 = CustomCNNBlock(64, 128)
        self.conv3 = CustomCNNBlock(128, 256)
        self.conv4 = CustomCNNBlock(256, 512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv4(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def convolution_demonstration():
    """Demonstrate convolution operations"""
    print("=== Convolution Operations ===")
    
    # Create sample input
    batch_size, channels, height, width = 1, 1, 5, 5
    input_tensor = torch.randn(batch_size, channels, height, width)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor:\n{input_tensor.squeeze()}")
    
    # Different convolution operations
    convolutions = {
        "3x3 conv, padding=1": nn.Conv2d(1, 1, kernel_size=3, padding=1),
        "3x3 conv, no padding": nn.Conv2d(1, 1, kernel_size=3, padding=0),
        "5x5 conv, padding=2": nn.Conv2d(1, 1, kernel_size=5, padding=2),
        "3x3 conv, stride=2": nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
    }
    
    for name, conv_layer in convolutions.items():
        output = conv_layer(input_tensor)
        print(f"\n{name}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Output:\n{output.squeeze().detach().numpy()}")
    
    return input_tensor

def pooling_demonstration():
    """Demonstrate pooling operations"""
    print("\n=== Pooling Operations ===")
    
    # Create sample feature map
    input_tensor = torch.randn(1, 1, 4, 4)
    print(f"Input tensor:\n{input_tensor.squeeze()}")
    
    # Different pooling operations
    pooling_ops = {
        "MaxPool2d(2,2)": nn.MaxPool2d(kernel_size=2, stride=2),
        "AvgPool2d(2,2)": nn.AvgPool2d(kernel_size=2, stride=2),
        "AdaptiveMaxPool2d(2,2)": nn.AdaptiveMaxPool2d((2, 2)),
        "AdaptiveAvgPool2d(1,1)": nn.AdaptiveAvgPool2d((1, 1)),
    }
    
    for name, pool_layer in pooling_ops.items():
        output = pool_layer(input_tensor)
        print(f"\n{name}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Output:\n{output.squeeze().detach().numpy()}")
    
    return input_tensor

def generate_synthetic_image_data():
    """Generate synthetic image data for examples"""
    print("\n=== Generating Synthetic Image Data ===")
    
    # Generate random images and simple labels
    n_samples = 1000
    height, width = 32, 32
    channels = 3
    n_classes = 5
    
    # Random images
    images = torch.randn(n_samples, channels, height, width)
    
    # Simple rule-based labels (based on mean pixel value)
    labels = []
    for img in images:
        mean_val = img.mean().item()
        if mean_val < -0.5:
            label = 0
        elif mean_val < -0.1:
            label = 1
        elif mean_val < 0.1:
            label = 2
        elif mean_val < 0.5:
            label = 3
        else:
            label = 4
        labels.append(label)
    
    labels = torch.tensor(labels)
    
    print(f"Generated {n_samples} images of shape {images[0].shape}")
    print(f"Label distribution: {torch.bincount(labels)}")
    
    return images, labels

def basic_cnn_training():
    """Train a basic CNN on synthetic data"""
    print("\n=== Basic CNN Training ===")
    
    # Generate data
    images, labels = generate_synthetic_image_data()
    
    # Split data
    train_size = int(0.8 * len(images))
    train_images, test_images = images[:train_size], images[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    model = BasicCNN(num_classes=5, input_channels=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for batch_images, batch_labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total_samples
        
        if epoch % 2 == 0:
            print(f"Epoch [{epoch+1}/10] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            outputs = model(batch_images)
            _, predicted = torch.max(outputs, 1)
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    return model

def feature_map_visualization():
    """Visualize feature maps from CNN layers"""
    print("\n=== Feature Map Visualization ===")
    
    # Create a simple model and input
    model = BasicCNN(num_classes=5, input_channels=3)
    sample_input = torch.randn(1, 3, 32, 32)
    
    # Hook to capture intermediate outputs
    feature_maps = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()
        return hook
    
    # Register hooks
    model.conv1.register_forward_hook(hook_fn('conv1'))
    model.conv2.register_forward_hook(hook_fn('conv2'))
    model.conv3.register_forward_hook(hook_fn('conv3'))
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_input)
    
    # Print feature map shapes
    print("Feature map shapes:")
    for name, feature_map in feature_maps.items():
        print(f"  {name}: {feature_map.shape}")
    
    # Show statistics of first few feature maps
    conv1_features = feature_maps['conv1'][0]  # Remove batch dimension
    print(f"\nConv1 feature statistics:")
    print(f"  Min: {conv1_features.min().item():.4f}")
    print(f"  Max: {conv1_features.max().item():.4f}")
    print(f"  Mean: {conv1_features.mean().item():.4f}")
    print(f"  Std: {conv1_features.std().item():.4f}")
    
    return feature_maps

def cnn_architecture_comparison():
    """Compare different CNN architectures"""
    print("\n=== CNN Architecture Comparison ===")
    
    # Create different models
    models = {
        'BasicCNN': BasicCNN(num_classes=10, input_channels=3),
        'LeNet5': LeNet5(num_classes=10),
        'ModernCNN': ModernCNN(num_classes=10, input_channels=3)
    }
    
    # Compare model characteristics
    sample_input_color = torch.randn(1, 3, 32, 32)
    sample_input_gray = torch.randn(1, 1, 32, 32)
    
    print("Model Comparison:")
    print("-" * 60)
    
    for name, model in models.items():
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"{name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        try:
            if name == 'LeNet5':
                # LeNet5 expects grayscale input
                with torch.no_grad():
                    output = model(sample_input_gray)
                print(f"  Output shape (grayscale input): {output.shape}")
            else:
                with torch.no_grad():
                    output = model(sample_input_color)
                print(f"  Output shape (color input): {output.shape}")
        except Exception as e:
            print(f"  Error in forward pass: {e}")
        
        print()
    
    return models

def data_augmentation_example():
    """Demonstrate data augmentation techniques"""
    print("\n=== Data Augmentation ===")
    
    # Define augmentation transforms
    augmentations = {
        'Original': transforms.Compose([transforms.ToTensor()]),
        
        'Random Flip': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ]),
        
        'Random Rotation': transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor()
        ]),
        
        'Color Jitter': transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor()
        ]),
        
        'Random Crop': transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.ToTensor()
        ]),
        
        'Multiple Augmentations': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    # Create dummy PIL image
    from PIL import Image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    
    print("Augmentation effects on tensor shapes and statistics:")
    
    for name, transform in augmentations.items():
        try:
            augmented = transform(dummy_image)
            print(f"{name}:")
            print(f"  Shape: {augmented.shape}")
            print(f"  Min: {augmented.min().item():.4f}")
            print(f"  Max: {augmented.max().item():.4f}")
            print(f"  Mean: {augmented.mean().item():.4f}")
            print()
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    return augmentations

def transfer_learning_basics():
    """Demonstrate transfer learning concepts"""
    print("\n=== Transfer Learning Basics ===")
    
    # Load a pretrained model (we'll use a simple example)
    # In practice, you'd use models like ResNet, VGG, etc.
    
    class PretrainedFeatureExtractor(nn.Module):
        """Simulated pretrained feature extractor"""
        
        def __init__(self):
            super(PretrainedFeatureExtractor, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
        
        def forward(self, x):
            return self.features(x)
    
    class TransferLearningModel(nn.Module):
        """Model using transfer learning"""
        
        def __init__(self, num_classes, freeze_features=True):
            super(TransferLearningModel, self).__init__()
            
            # Pretrained feature extractor
            self.feature_extractor = PretrainedFeatureExtractor()
            
            # Freeze feature extractor if specified
            if freeze_features:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
            
            # Custom classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256 * 4 * 4, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
            return self.classifier(features)
    
    # Compare frozen vs unfrozen feature extractor
    models = {
        'Frozen Features': TransferLearningModel(num_classes=5, freeze_features=True),
        'Unfrozen Features': TransferLearningModel(num_classes=5, freeze_features=False)
    }
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"{name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")
        print()
    
    return models

def cnn_receptive_field_analysis():
    """Analyze receptive field of CNN layers"""
    print("\n=== Receptive Field Analysis ===")
    
    def calculate_receptive_field(layers):
        """Calculate receptive field size through layers"""
        rf_size = 1
        jump = 1
        
        print("Layer-by-layer receptive field analysis:")
        print("Layer | RF Size | Jump | Description")
        print("-" * 50)
        
        for i, (layer_type, kernel_size, stride, padding) in enumerate(layers):
            if layer_type in ['conv', 'pool']:
                # Update receptive field
                rf_size = rf_size + (kernel_size - 1) * jump
                jump = jump * stride
                
                print(f"{i+1:5} | {rf_size:7} | {jump:4} | {layer_type} k={kernel_size}, s={stride}, p={padding}")
        
        return rf_size
    
    # Analyze BasicCNN architecture
    basic_cnn_layers = [
        ('conv', 3, 1, 1),    # conv1
        ('pool', 2, 2, 0),    # pool1
        ('conv', 3, 1, 1),    # conv2
        ('pool', 2, 2, 0),    # pool2
        ('conv', 3, 1, 1),    # conv3
        ('pool', 2, 2, 0),    # pool3
    ]
    
    print("BasicCNN Receptive Field Analysis:")
    final_rf = calculate_receptive_field(basic_cnn_layers)
    print(f"\nFinal receptive field size: {final_rf}x{final_rf}")
    
    return final_rf

def main():
    """Run all examples"""
    print("PyTorch CNN Basics Tutorial")
    print("=" * 40)
    
    convolution_demonstration()
    pooling_demonstration()
    generate_synthetic_image_data()
    basic_cnn_training()
    feature_map_visualization()
    cnn_architecture_comparison()
    data_augmentation_example()
    transfer_learning_basics()
    cnn_receptive_field_analysis()
    
    print("\n" + "=" * 40)
    print("Tutorial completed successfully!")

if __name__ == "__main__":
    main()