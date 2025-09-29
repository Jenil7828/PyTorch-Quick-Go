"""
PyTorch Basics: Automatic Differentiation (Autograd)
===================================================

This file covers PyTorch's automatic differentiation system:
- Understanding gradients and requires_grad
- Forward and backward passes
- Gradient computation
- Computational graphs
- Common gradient operations
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def basic_autograd():
    """Introduction to autograd and gradient computation"""
    print("=== Basic Autograd Examples ===")
    
    # Create a tensor with gradient tracking
    x = torch.tensor(2.0, requires_grad=True)
    print(f"x = {x}, requires_grad = {x.requires_grad}")
    
    # Perform operations
    y = x**2 + 3*x + 1
    print(f"y = x² + 3x + 1 = {y}")
    
    # Compute gradients
    y.backward()
    print(f"dy/dx = {x.grad}")
    print(f"Expected: 2x + 3 = 2(2) + 3 = 7")
    
    # Multiple computations require zero_grad()
    x.grad.zero_()  # Clear previous gradients
    z = 2*x**3 - x
    z.backward()
    print(f"For z = 2x³ - x, dz/dx = {x.grad}")
    print(f"Expected: 6x² - 1 = 6(4) - 1 = 23")
    
    return x, y, z

def vector_gradients():
    """Working with vector gradients"""
    print("\n=== Vector Gradients ===")
    
    # Vector input
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    print(f"x = {x}")
    
    # Scalar output (sum of squares)
    y = torch.sum(x**2)
    print(f"y = sum(x²) = {y}")
    
    y.backward()
    print(f"dy/dx = {x.grad}")
    print(f"Expected: [2x₁, 2x₂, 2x₃] = [2, 4, 6]")
    
    # Reset gradients
    x.grad.zero_()
    
    # Vector output requires gradient argument
    y_vector = x**2
    gradient = torch.ones_like(y_vector)  # Gradient for each output
    y_vector.backward(gradient)
    print(f"For vector output y = x², dy/dx = {x.grad}")
    
    return x

def computational_graph():
    """Understanding the computational graph"""
    print("\n=== Computational Graph ===")
    
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    
    # Build computational graph
    c = a * b
    d = c + a
    e = d**2
    
    print(f"a = {a}, b = {b}")
    print(f"c = a * b = {c}")
    print(f"d = c + a = {d}")
    print(f"e = d² = {e}")
    
    # Backward pass
    e.backward()
    
    print(f"∂e/∂a = {a.grad}")
    print(f"∂e/∂b = {b.grad}")
    
    # Manual calculation verification
    # e = (a*b + a)² = (2*3 + 2)² = 8² = 64
    # ∂e/∂a = 2(a*b + a) * (b + 1) = 2*8*4 = 64
    # ∂e/∂b = 2(a*b + a) * a = 2*8*2 = 32
    
    return a, b, e

def gradient_control():
    """Controlling gradient computation"""
    print("\n=== Gradient Control ===")
    
    x = torch.randn(3, requires_grad=True)
    print(f"x = {x}")
    
    # Disable gradient computation temporarily
    with torch.no_grad():
        y = x**2
        print(f"y computed with no_grad: {y}")
        print(f"y.requires_grad: {y.requires_grad}")
    
    # Detach from computational graph
    y_detached = (x**2).detach()
    print(f"y_detached.requires_grad: {y_detached.requires_grad}")
    
    # Stop gradient flow at specific point
    z = x**2
    z_stopped = z.detach()
    w = z_stopped + x
    w.sum().backward()
    print(f"Gradient after stopping: {x.grad}")
    
    return x

def higher_order_gradients():
    """Computing higher-order gradients"""
    print("\n=== Higher-Order Gradients ===")
    
    x = torch.tensor(2.0, requires_grad=True)
    
    # First derivative
    y = x**3
    dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"y = x³, dy/dx = {dy_dx}")
    
    # Second derivative
    d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
    print(f"d²y/dx² = {d2y_dx2}")
    print(f"Expected: d/dx(3x²) = 6x = 6(2) = 12")
    
    return x, dy_dx, d2y_dx2

def gradient_optimization_example():
    """Simple optimization using gradients"""
    print("\n=== Gradient-Based Optimization ===")
    
    # Find minimum of f(x) = (x - 3)² + 1
    x = torch.tensor(0.0, requires_grad=True)
    learning_rate = 0.1
    losses = []
    
    print("Optimizing f(x) = (x - 3)² + 1")
    print("Target minimum: x = 3, f(3) = 1")
    
    for i in range(50):
        # Forward pass
        loss = (x - 3)**2 + 1
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        with torch.no_grad():
            x -= learning_rate * x.grad
        
        # Clear gradients
        x.grad.zero_()
        
        if i % 10 == 0:
            print(f"Step {i}: x = {x.item():.4f}, loss = {loss.item():.4f}")
    
    print(f"Final: x = {x.item():.4f}, loss = {loss.item():.4f}")
    
    return x, losses

def jacobian_example():
    """Computing Jacobian matrices"""
    print("\n=== Jacobian Matrix ===")
    
    def func(x):
        """Vector function: f(x) = [x₁², x₁*x₂, x₂²]"""
        return torch.stack([x[0]**2, x[0]*x[1], x[1]**2])
    
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    
    # Compute Jacobian using autograd
    y = func(x)
    jacobian = torch.zeros(3, 2)
    
    for i in range(3):
        grad_outputs = torch.zeros(3)
        grad_outputs[i] = 1
        grads = torch.autograd.grad(y, x, grad_outputs=grad_outputs, retain_graph=True)[0]
        jacobian[i] = grads
    
    print(f"Input: x = {x}")
    print(f"Output: f(x) = {y}")
    print(f"Jacobian matrix:\n{jacobian}")
    
    # Expected Jacobian:
    # ∂f₁/∂x₁ = 2x₁ = 4, ∂f₁/∂x₂ = 0
    # ∂f₂/∂x₁ = x₂ = 3,  ∂f₂/∂x₂ = x₁ = 2
    # ∂f₃/∂x₁ = 0,       ∂f₃/∂x₂ = 2x₂ = 6
    
    return jacobian

def practical_neural_network_gradients():
    """Gradients in a simple neural network"""
    print("\n=== Neural Network Gradients ===")
    
    # Simple linear layer
    layer = nn.Linear(2, 1)
    print(f"Initial weights: {layer.weight}")
    print(f"Initial bias: {layer.bias}")
    
    # Input data
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0], [0.0]])
    
    # Forward pass
    output = layer(x)
    loss = nn.MSELoss()(output, target)
    
    print(f"Output: {output}")
    print(f"Loss: {loss}")
    
    # Backward pass
    loss.backward()
    
    print(f"Weight gradients: {layer.weight.grad}")
    print(f"Bias gradients: {layer.bias.grad}")
    
    return layer, loss

def main():
    """Run all examples"""
    print("PyTorch Autograd Tutorial")
    print("=" * 40)
    
    basic_autograd()
    vector_gradients()
    computational_graph()
    gradient_control()
    higher_order_gradients()
    gradient_optimization_example()
    jacobian_example()
    practical_neural_network_gradients()
    
    print("\n" + "=" * 40)
    print("Tutorial completed successfully!")

if __name__ == "__main__":
    main()