"""
PyTorch Tensors Basics Tutorial
===============================

This tutorial covers the fundamental concepts of PyTorch tensors, including:
- Installation and importing
- GPU availability checking
- Tensor creation methods
- Data types and conversions
- Mathematical operations
- In-place operations
- GPU training basics
- Dimension manipulation
- NumPy interoperability

Tensors are specialized multi-dimensional arrays designed for mathematical 
and computational efficiency in deep learning.
"""

# ============================================================================
# 1. INSTALLATION AND IMPORTING
# ============================================================================

# Install PyTorch (uncomment if needed)
# !pip install torch

# Import PyTorch
import torch
print("PyTorch version:", torch.__version__)

# ============================================================================
# 2. CHECKING GPU AVAILABILITY
# ============================================================================

# Check if CUDA GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using GPU:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU.")

print("Current device:", device)

# ============================================================================
# 3. TENSOR CREATION FROM DATA
# ============================================================================

# Creating tensors from lists
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print("\nTensor from data:")
print(x_data)

# Creating tensors with specific shapes and values
# Empty tensor (uninitialized data)
x_empty = torch.empty(3, 4)
print("\nEmpty tensor:")
print(x_empty)

# Zeros tensor
x_zeros = torch.zeros(3, 4)
print("\nZeros tensor:")
print(x_zeros)

# Ones tensor
x_ones = torch.ones(3, 4)
print("\nOnes tensor:")
print(x_ones)

# Random tensor
x_random = torch.rand(3, 4)
print("\nRandom tensor:")
print(x_random)

# Random normal distribution
x_randn = torch.randn(3, 4)
print("\nRandom normal tensor:")
print(x_randn)

# Identity matrix
x_eye = torch.eye(3)
print("\nIdentity tensor:")
print(x_eye)

# Tensor with range
x_range = torch.arange(0, 10, 2)
print("\nRange tensor:")
print(x_range)

# Linspace tensor
x_linspace = torch.linspace(0, 1, 5)
print("\nLinspace tensor:")
print(x_linspace)

# ============================================================================
# 4. TENSOR CREATION FROM EXISTING TENSORS
# ============================================================================

# Original tensor
original = torch.rand(3, 4)
print("\nOriginal tensor:")
print(original)

# Create new tensors with same shape but different content
x_zeros_like = torch.zeros_like(original)
print("\nZeros like original:")
print(x_zeros_like)

x_ones_like = torch.ones_like(original)
print("\nOnes like original:")
print(x_ones_like)

x_rand_like = torch.rand_like(original)
print("\nRandom like original:")
print(x_rand_like)

# Override data type (using float16 instead of int32 for rand_like)
x_rand_like_float16 = torch.rand_like(original, dtype=torch.float16)
print("\nRandom like original (float16):")
print(x_rand_like_float16)

# ============================================================================
# 5. TENSOR PROPERTIES
# ============================================================================

tensor = torch.rand(3, 4)
print("\nTensor properties:")
print("Tensor:", tensor)
print("Shape:", tensor.shape)
print("Size:", tensor.size())
print("Data type:", tensor.dtype)
print("Device:", tensor.device)
print("Number of dimensions:", tensor.ndim)
print("Number of elements:", tensor.numel())

# ============================================================================
# 6. DATA TYPES AND CONVERSION
# ============================================================================

# Different data types
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
double_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

print("\nData types:")
print("Integer tensor:", int_tensor, "dtype:", int_tensor.dtype)
print("Float tensor:", float_tensor, "dtype:", float_tensor.dtype)
print("Double tensor:", double_tensor, "dtype:", double_tensor.dtype)

# Type conversion
float_to_int = float_tensor.to(torch.int32)
print("\nType conversion (to method):")
print("Float to int:", float_to_int, "dtype:", float_to_int.dtype)

# Alternative type conversion methods
float_to_int_alt = float_tensor.int()
print("Float to int (int method):", float_to_int_alt, "dtype:", float_to_int_alt.dtype)

int_to_float = int_tensor.float()
print("Int to float:", int_to_float, "dtype:", int_to_float.dtype)

# ============================================================================
# 7. MATHEMATICAL OPERATIONS
# ============================================================================

# Element-wise operations
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print("\nMathematical Operations:")
print("x:", x)
print("y:", y)

# Addition
add_result1 = torch.add(x, y)
add_result2 = x + y
print("\nAddition:")
print("torch.add(x, y):", add_result1)
print("x + y:", add_result2)

# Subtraction
sub_result1 = torch.sub(x, y)
sub_result2 = x - y
print("\nSubtraction:")
print("torch.sub(x, y):", sub_result1)
print("x - y:", sub_result2)

# Multiplication (element-wise)
mul_result1 = torch.mul(x, y)
mul_result2 = x * y
print("\nElement-wise multiplication:")
print("torch.mul(x, y):", mul_result1)
print("x * y:", mul_result2)

# Division
div_result1 = torch.div(x, y)
div_result2 = x / y
print("\nDivision:")
print("torch.div(x, y):", div_result1)
print("x / y:", div_result2)

# Power
pow_result1 = torch.pow(x, 2)
pow_result2 = x ** 2
print("\nPower (square):")
print("torch.pow(x, 2):", pow_result1)
print("x ** 2:", pow_result2)

# Matrix multiplication
a = torch.randn(2, 3)
b = torch.randn(3, 4)
matmul_result1 = torch.matmul(a, b)
matmul_result2 = a @ b
print("\nMatrix multiplication:")
print("a shape:", a.shape, "b shape:", b.shape)
print("torch.matmul(a, b) shape:", matmul_result1.shape)
print("a @ b shape:", matmul_result2.shape)

# Scalar operations
scalar = 5
x_scalar = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print("\nScalar operations:")
print("Original tensor:", x_scalar)
print("Add scalar:", x_scalar + scalar)
print("Multiply by scalar:", x_scalar * scalar)
print("Divide by scalar:", x_scalar / scalar)

# ============================================================================
# 8. IN-PLACE OPERATIONS
# ============================================================================

print("\nIn-place operations:")
x_inplace = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y_inplace = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print("Before in-place operations:")
print("x:", x_inplace)
print("y:", y_inplace)

# In-place addition (modifies x)
x_inplace.add_(y_inplace)
print("\nAfter x.add_(y):")
print("x:", x_inplace)

# In-place subtraction
x_inplace.sub_(2)
print("\nAfter x.sub_(2):")
print("x:", x_inplace)

# In-place multiplication
x_inplace.mul_(2)
print("\nAfter x.mul_(2):")
print("x:", x_inplace)

# Note: In-place operations save memory but modify the original tensor

# ============================================================================
# 9. TRAINING ON GPU (GPU OPERATIONS)
# ============================================================================

print("\nGPU Operations:")
if torch.cuda.is_available():
    # Move tensors to GPU
    x_gpu = torch.randn(3, 4).to(device)
    y_gpu = torch.randn(3, 4).to(device)
    
    print("Tensors moved to GPU:")
    print("x_gpu device:", x_gpu.device)
    print("y_gpu device:", y_gpu.device)
    
    # Operations on GPU
    result_gpu = x_gpu + y_gpu
    print("GPU operation result device:", result_gpu.device)
    
    # Move back to CPU for printing
    result_cpu = result_gpu.cpu()
    print("Result moved back to CPU:", result_cpu.device)
    
    # Direct GPU tensor creation
    gpu_tensor = torch.randn(2, 3, device=device)
    print("Direct GPU tensor device:", gpu_tensor.device)
else:
    print("GPU not available, skipping GPU operations")
    # CPU operations as fallback
    x_cpu = torch.randn(3, 4)
    y_cpu = torch.randn(3, 4)
    result_cpu = x_cpu + y_cpu
    print("CPU operation completed on device:", result_cpu.device)

# ============================================================================
# 10. PLAYING WITH DIMENSIONS
# ============================================================================

print("\nDimension operations:")
x_dim = torch.randn(2, 3, 4)
print("Original tensor shape:", x_dim.shape)

# Reshape
reshaped = x_dim.view(3, 8)
print("Reshaped (view):", reshaped.shape)

# Another reshape method
reshaped2 = x_dim.reshape(4, 6)
print("Reshaped (reshape):", reshaped2.shape)

# Squeeze (remove dimensions of size 1)
x_squeeze = torch.randn(1, 3, 1, 4, 1)
print("Before squeeze:", x_squeeze.shape)
squeezed = x_squeeze.squeeze()
print("After squeeze:", squeezed.shape)

# Unsqueeze (add dimensions of size 1)
x_unsqueeze = torch.randn(3, 4)
print("Before unsqueeze:", x_unsqueeze.shape)
unsqueezed = x_unsqueeze.unsqueeze(0)  # Add dimension at index 0
print("After unsqueeze(0):", unsqueezed.shape)
unsqueezed2 = x_unsqueeze.unsqueeze(-1)  # Add dimension at the end
print("After unsqueeze(-1):", unsqueezed2.shape)

# Transpose
x_transpose = torch.randn(3, 4)
print("Original shape:", x_transpose.shape)
transposed = x_transpose.t()  # 2D transpose
print("Transposed shape:", transposed.shape)

# Permute (generalized transpose)
x_permute = torch.randn(2, 3, 4)
print("Before permute:", x_permute.shape)
permuted = x_permute.permute(2, 0, 1)  # Rearrange dimensions
print("After permute(2, 0, 1):", permuted.shape)

# Concatenation
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3)
print("Tensors to concatenate:", x1.shape, x2.shape)

concat_dim0 = torch.cat([x1, x2], dim=0)  # Concatenate along dimension 0
print("Concatenated along dim 0:", concat_dim0.shape)

concat_dim1 = torch.cat([x1, x2], dim=1)  # Concatenate along dimension 1
print("Concatenated along dim 1:", concat_dim1.shape)

# Stacking
stacked = torch.stack([x1, x2], dim=0)  # Create new dimension
print("Stacked shape:", stacked.shape)

# ============================================================================
# 11. NUMPY VS PYTORCH INTEROPERABILITY
# ============================================================================

print("\nNumPy vs PyTorch interoperability:")

# Note: NumPy might not be available in all environments
try:
    import numpy as np
    
    # NumPy array to PyTorch tensor
    numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
    tensor_from_numpy = torch.from_numpy(numpy_array)
    print("NumPy array:")
    print(numpy_array)
    print("PyTorch tensor from NumPy:")
    print(tensor_from_numpy)
    print("Tensor dtype:", tensor_from_numpy.dtype)
    
    # PyTorch tensor to NumPy array
    pytorch_tensor = torch.randn(2, 3)
    numpy_from_tensor = pytorch_tensor.numpy()
    print("\nPyTorch tensor:")
    print(pytorch_tensor)
    print("NumPy array from PyTorch:")
    print(numpy_from_tensor)
    
    # Important note about memory sharing
    print("\nMemory sharing demonstration:")
    shared_tensor = torch.from_numpy(numpy_array)
    print("Original NumPy array:", numpy_array)
    shared_tensor[0, 0] = 999  # Modify tensor
    print("NumPy array after tensor modification:", numpy_array)
    print("They share memory!")
    
    # To avoid memory sharing, use .clone()
    independent_tensor = torch.from_numpy(numpy_array).clone()
    independent_tensor[0, 1] = 888
    print("NumPy array after independent tensor modification:", numpy_array)
    print("Independent tensor:", independent_tensor)
    
except ImportError:
    print("NumPy not available. Skipping NumPy interoperability examples.")
    print("Install NumPy with: pip install numpy")
    
    # Alternative: Create similar examples with pure PyTorch
    print("Alternative: Converting between different tensor types")
    float_tensor = torch.tensor([1.0, 2.0, 3.0])
    int_tensor = float_tensor.to(torch.int32)
    print("Float tensor:", float_tensor)
    print("Int tensor:", int_tensor)

# ============================================================================
# 12. SUMMARY AND BEST PRACTICES
# ============================================================================

print("\n" + "="*60)
print("PYTORCH TENSORS BASICS SUMMARY")
print("="*60)
print("1. Always check device compatibility (CPU/GPU)")
print("2. Be aware of tensor data types for computation efficiency")
print("3. Use appropriate tensor creation methods for your use case")
print("4. In-place operations save memory but modify original tensors")
print("5. GPU tensors require explicit device management")
print("6. Dimension operations are crucial for neural network operations")
print("7. NumPy interoperability enables seamless data science workflows")
print("8. Always consider memory implications when working with large tensors")
print("="*60)

print("\nTutorial completed successfully!")