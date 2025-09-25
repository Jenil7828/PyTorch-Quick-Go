"""
PyTorch Basics: Tensors and Basic Operations
============================================

This file covers fundamental tensor operations in PyTorch, including:
- Creating tensors
- Basic operations
- Indexing and slicing
- Reshaping and manipulation
- Data types and device management
"""

import torch
import numpy as np

def tensor_creation_examples():
    """Demonstrates various ways to create tensors in PyTorch"""
    print("=== Tensor Creation Examples ===")
    
    # Create from data
    data = [[1, 2], [3, 4]]
    tensor_from_data = torch.tensor(data)
    print(f"From list: {tensor_from_data}")
    
    # From numpy array
    np_array = np.array(data)
    tensor_from_numpy = torch.from_numpy(np_array)
    print(f"From numpy: {tensor_from_numpy}")
    
    # Zeros and ones
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    print(f"Zeros:\n{zeros}")
    print(f"Ones:\n{ones}")
    
    # Random tensors
    random_tensor = torch.rand(2, 3)  # Uniform [0, 1)
    normal_tensor = torch.randn(2, 3)  # Normal distribution
    print(f"Random uniform:\n{random_tensor}")
    print(f"Random normal:\n{normal_tensor}")
    
    # Identity matrix
    identity = torch.eye(3)
    print(f"Identity matrix:\n{identity}")
    
    # Filled with specific value
    filled = torch.full((2, 3), 7.5)
    print(f"Filled with 7.5:\n{filled}")
    
    return tensor_from_data

def tensor_properties_and_attributes():
    """Explores tensor properties and attributes"""
    print("\n=== Tensor Properties ===")
    
    tensor = torch.randn(3, 4, 5)
    print(f"Shape: {tensor.shape}")
    print(f"Size: {tensor.size()}")
    print(f"Data type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Number of dimensions: {tensor.ndim}")
    print(f"Total elements: {tensor.numel()}")
    
    return tensor

def basic_operations():
    """Demonstrates basic tensor operations"""
    print("\n=== Basic Operations ===")
    
    # Create sample tensors
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    
    print(f"Tensor a:\n{a}")
    print(f"Tensor b:\n{b}")
    
    # Arithmetic operations
    print(f"Addition: a + b =\n{a + b}")
    print(f"Subtraction: a - b =\n{a - b}")
    print(f"Element-wise multiplication: a * b =\n{a * b}")
    print(f"Element-wise division: a / b =\n{a / b}")
    print(f"Power: a ** 2 =\n{a ** 2}")
    
    # Matrix operations
    print(f"Matrix multiplication: a @ b =\n{a @ b}")
    print(f"Transpose: a.T =\n{a.T}")
    
    # In-place operations (end with _)
    c = a.clone()  # Make a copy
    c.add_(1)  # Add 1 in-place
    print(f"After in-place addition of 1:\n{c}")
    
    return a, b

def indexing_and_slicing():
    """Demonstrates tensor indexing and slicing"""
    print("\n=== Indexing and Slicing ===")
    
    tensor = torch.arange(12).reshape(3, 4)
    print(f"Original tensor:\n{tensor}")
    
    # Basic indexing
    print(f"Element at [1, 2]: {tensor[1, 2]}")
    print(f"First row: {tensor[0, :]}")
    print(f"Last column: {tensor[:, -1]}")
    
    # Slicing
    print(f"First two rows, first three columns:\n{tensor[:2, :3]}")
    print(f"Every other element in first row: {tensor[0, ::2]}")
    
    # Boolean indexing
    mask = tensor > 5
    print(f"Boolean mask (elements > 5):\n{mask}")
    print(f"Elements > 5: {tensor[mask]}")
    
    # Advanced indexing
    indices = torch.tensor([0, 2])
    print(f"Rows 0 and 2:\n{tensor[indices]}")
    
    return tensor

def reshaping_and_manipulation():
    """Demonstrates tensor reshaping and manipulation"""
    print("\n=== Reshaping and Manipulation ===")
    
    tensor = torch.arange(12)
    print(f"Original tensor: {tensor}")
    
    # Reshape
    reshaped = tensor.reshape(3, 4)
    print(f"Reshaped to 3x4:\n{reshaped}")
    
    # View (shares memory)
    viewed = tensor.view(2, 6)
    print(f"Viewed as 2x6:\n{viewed}")
    
    # Flatten
    flattened = reshaped.flatten()
    print(f"Flattened: {flattened}")
    
    # Squeeze and unsqueeze
    tensor_with_singleton = torch.randn(1, 3, 1, 4)
    print(f"Original shape: {tensor_with_singleton.shape}")
    squeezed = tensor_with_singleton.squeeze()
    print(f"After squeeze: {squeezed.shape}")
    unsqueezed = squeezed.unsqueeze(0)
    print(f"After unsqueeze(0): {unsqueezed.shape}")
    
    # Concatenation
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    concatenated = torch.cat([a, b], dim=0)  # Along rows
    print(f"Concatenated along dim 0: {concatenated.shape}")
    
    # Stack
    stacked = torch.stack([a, b], dim=0)
    print(f"Stacked along dim 0: {stacked.shape}")
    
    return tensor

def data_types_and_conversion():
    """Demonstrates tensor data types and conversions"""
    print("\n=== Data Types and Conversion ===")
    
    # Different data types
    float_tensor = torch.tensor([1.0, 2.0, 3.0])
    int_tensor = torch.tensor([1, 2, 3])
    bool_tensor = torch.tensor([True, False, True])
    
    print(f"Float tensor dtype: {float_tensor.dtype}")
    print(f"Int tensor dtype: {int_tensor.dtype}")
    print(f"Bool tensor dtype: {bool_tensor.dtype}")
    
    # Type conversion
    float_to_int = float_tensor.int()
    int_to_float = int_tensor.float()
    
    print(f"Float to int: {float_to_int} (dtype: {float_to_int.dtype})")
    print(f"Int to float: {int_to_float} (dtype: {int_to_float.dtype})")
    
    # Specify dtype during creation
    specific_dtype = torch.tensor([1, 2, 3], dtype=torch.float64)
    print(f"Specific dtype: {specific_dtype.dtype}")
    
    return float_tensor

def device_management():
    """Demonstrates device management (CPU/GPU)"""
    print("\n=== Device Management ===")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create tensor on specific device
    tensor_cpu = torch.randn(3, 3)
    print(f"CPU tensor device: {tensor_cpu.device}")
    
    if torch.cuda.is_available():
        tensor_gpu = torch.randn(3, 3, device="cuda")
        print(f"GPU tensor device: {tensor_gpu.device}")
        
        # Move tensor between devices
        tensor_moved_to_gpu = tensor_cpu.to(device)
        tensor_moved_to_cpu = tensor_moved_to_gpu.cpu()
        print(f"Moved to GPU: {tensor_moved_to_gpu.device}")
        print(f"Moved back to CPU: {tensor_moved_to_cpu.device}")
    else:
        print("CUDA not available, using CPU only")
    
    return tensor_cpu

def main():
    """Run all examples"""
    print("PyTorch Tensor Basics Tutorial")
    print("=" * 40)
    
    tensor_creation_examples()
    tensor_properties_and_attributes()
    basic_operations()
    indexing_and_slicing()
    reshaping_and_manipulation()
    data_types_and_conversion()
    device_management()
    
    print("\n" + "=" * 40)
    print("Tutorial completed successfully!")

if __name__ == "__main__":
    main()