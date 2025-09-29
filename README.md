# PyTorch-Quick-Go 🚀

A comprehensive repository for on-the-go revision and learning of PyTorch, covering everything from basic tensor operations to advanced deep learning concepts in NLP and Computer Vision.

## 📚 Table of Contents

### 01. Basics
- **[Tensor Basics](01_basics/01_tensors_basics.py)** - Tensor creation, operations, indexing, and device management
- **[Autograd & Gradients](01_basics/02_autograd_gradients.py)** - Automatic differentiation, computational graphs, and gradient computation
- **[Data Loading](01_basics/03_data_loading.py)** - Datasets, DataLoaders, custom datasets, and data preprocessing

### 02. Intermediate
- **[Neural Networks](02_intermediate/01_neural_networks.py)** - Building neural networks, layers, activations, and model architecture
- **[Optimization](02_intermediate/02_optimization.py)** - Optimizers, learning rate scheduling, regularization, and training best practices

### 03. Advanced
- **[Custom Layers](03_advanced/01_custom_layers.py)** - Creating custom layers, modules, and advanced architectures

### 04. Natural Language Processing (NLP)
- **[Text Processing](04_nlp/01_text_processing.py)** - Text preprocessing, tokenization, vocabulary building, and text datasets
- **[RNN/LSTM/GRU](04_nlp/02_rnn_lstm_gru.py)** - Recurrent networks, sequence modeling, and sequence-to-sequence models

### 05. Computer Vision
- **[CNN Basics](05_computer_vision/01_cnn_basics.py)** - Convolutional networks, pooling, architectures, and transfer learning

### 06. Projects
- **[Mini Projects](06_projects/)** - Complete end-to-end projects combining multiple concepts

## 🛠️ Setup

### Requirements
Install the required packages using:

```bash
pip install -r requirements.txt
```

### Main Dependencies
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- transformers >= 4.20.0

## 🚀 Quick Start

Each file is self-contained and can be run independently:

```bash
# Run tensor basics examples
python 01_basics/01_tensors_basics.py

# Run CNN examples
python 05_computer_vision/01_cnn_basics.py

# Run any specific module
python <path_to_file>.py
```

## 📖 Learning Path

### Beginner Path
1. Start with **Tensor Basics** to understand PyTorch fundamentals
2. Learn **Autograd & Gradients** for automatic differentiation
3. Master **Data Loading** for handling datasets
4. Build your first **Neural Networks**

### Intermediate Path
1. Dive into **Optimization** techniques
2. Explore **Custom Layers** for advanced architectures
3. Choose your domain: **NLP** or **Computer Vision**

### Advanced Path
1. Work through domain-specific advanced topics
2. Complete the **Mini Projects** to integrate concepts
3. Experiment with the code and build your own projects

## 💡 Key Features

### Comprehensive Coverage
- **Basics**: Tensors, autograd, data loading
- **Intermediate**: Neural networks, optimization
- **Advanced**: Custom layers, domain-specific topics
- **Practical**: Real-world examples and projects

### Educational Design
- **Self-contained modules**: Each file runs independently
- **Progressive complexity**: Builds from basics to advanced
- **Extensive comments**: Clear explanations throughout
- **Practical examples**: Real-world applicable code

### Domain Specific
- **NLP**: Text processing, RNNs, transformers
- **Computer Vision**: CNNs, image processing, architectures
- **General ML**: Optimization, regularization, best practices

## 🎯 What You'll Learn

### Core PyTorch
- Tensor operations and manipulations
- Automatic differentiation and gradient computation
- Building and training neural networks
- Data loading and preprocessing pipelines

### Deep Learning Fundamentals
- Neural network architectures
- Optimization algorithms and techniques
- Regularization and best practices
- Model evaluation and debugging

### NLP with PyTorch
- Text preprocessing and tokenization
- Recurrent neural networks (RNN, LSTM, GRU)
- Sequence-to-sequence models
- Attention mechanisms and transformers

### Computer Vision with PyTorch
- Convolutional neural networks
- Image preprocessing and augmentation
- Popular CNN architectures
- Transfer learning techniques

## 🔧 Code Structure

Each module follows a consistent structure:
- **Imports and setup**
- **Class/function definitions**
- **Demonstration functions**
- **Main execution with examples**
- **Comprehensive documentation**

Example structure:
```python
"""
Module: Description
==================

Learning objectives and contents overview
"""

# Imports
import torch
import torch.nn as nn

# Class definitions
class ExampleModel(nn.Module):
    # Implementation with detailed comments
    pass

# Demonstration functions
def demonstrate_concept():
    # Clear examples with explanations
    pass

def main():
    # Run all demonstrations
    pass

if __name__ == "__main__":
    main()
```

## 🤝 Contributing

This repository is designed for learning and revision. Feel free to:
- Report issues or bugs
- Suggest improvements
- Add new examples or explanations
- Share your learning experience

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🎓 Additional Resources

### PyTorch Documentation
- [Official PyTorch Docs](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Learning Materials
- Deep Learning with PyTorch (Book)
- Fast.ai Deep Learning Course
- CS231n: Convolutional Neural Networks

### Community
- [PyTorch Forums](https://discuss.pytorch.org/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)

---

**Happy Learning! 🎉**

Start your PyTorch journey today and master deep learning step by step!
