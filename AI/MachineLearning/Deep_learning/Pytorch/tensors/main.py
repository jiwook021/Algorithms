import torch
import numpy as np

def tensor_basics_tutorial():
    """
    Demonstrates PyTorch tensor creation, operations, and manipulations.
    Time Complexity: O(n) for most operations where n is the number of elements
    Memory Complexity: O(n) for tensor storage
    """
    # 1. Creating tensors (different methods)
    print("===== TENSOR CREATION =====")
    
    # From Python lists
    data_list = [[1, 2, 3], [4, 5, 6]]
    tensor_from_list = torch.tensor(data_list)
    print(f"From list: \n{tensor_from_list}")
    
    # From NumPy arrays (zero memory copy when on CPU)
    np_array = np.array(data_list)
    tensor_from_numpy = torch.from_numpy(np_array)
    print(f"From NumPy: \n{tensor_from_numpy}")
    
    # Common initialization functions
    zeros = torch.zeros(2, 3)  # 2x3 tensor of zeros
    ones = torch.ones(2, 3)    # 2x3 tensor of ones
    rand = torch.rand(2, 3)    # 2x3 tensor of random values [0, 1)
    randn = torch.randn(2, 3)  # 2x3 tensor from standard normal distribution
    
    print(f"Zeros: \n{zeros}")
    print(f"Ones: \n{ones}")
    print(f"Random [0,1): \n{rand}")
    print(f"Random Normal: \n{randn}")
    
    # Range initialization
    range_tensor = torch.arange(0, 10, step=2)  # [0, 2, 4, 6, 8]
    print(f"Range: \n{range_tensor}")
    
    # 2. Tensor attributes
    print("\n===== TENSOR ATTRIBUTES =====")
    x = torch.randn(3, 4, 5)
    print(f"Shape: {x.shape}")        # Size of each dimension
    print(f"Rank: {x.ndim}")          # Number of dimensions
    print(f"Datatype: {x.dtype}")     # Data type
    print(f"Device: {x.device}")      # CPU/GPU
    print(f"Total elements: {x.numel()}")  # Number of elements
    
    # 3. Basic operations
    print("\n===== BASIC OPERATIONS =====")
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    
    # Element-wise operations
    print(f"a + b: {a + b}")               # Addition
    print(f"a - b: {a - b}")               # Subtraction
    print(f"a * b: {a * b}")               # Element-wise multiplication
    print(f"a / b: {a / b}")               # Division
    
    # Mathematical functions
    print(f"exp(a): {torch.exp(a)}")       # Exponential
    print(f"log(a): {torch.log(torch.abs(a))}")  # Natural logarithm
    print(f"sin(a): {torch.sin(a)}")       # Sine
    
    # 4. Indexing and slicing (similar to NumPy)
    print("\n===== INDEXING AND SLICING =====")
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Original matrix: \n{matrix}")
    
    print(f"First row: {matrix[0]}")
    print(f"First column: {matrix[:, 0]}")
    print(f"Submatrix (first 2 rows, last 2 columns): \n{matrix[:2, 1:]}")
    
    # Advanced indexing
    indices = torch.tensor([0, 2])  # Select first and third rows
    print(f"Selected rows: \n{matrix[indices]}")
    
    # Boolean indexing
    mask = matrix > 5
    print(f"Values > 5: {matrix[mask]}")
    
    # 5. Reshaping operations
    print("\n===== RESHAPING =====")
    tensor = torch.arange(12)
    print(f"Original: {tensor}")
    
    # Reshape
    reshaped = tensor.reshape(3, 4)  # or tensor.view(3, 4)
    print(f"Reshaped to 3x4: \n{reshaped}")
    
    # Transpose
    transposed = reshaped.t()  # or reshaped.transpose(0, 1)
    print(f"Transposed: \n{transposed}")
    
    # Permute dimensions (for tensors with more dimensions)
    three_dim = tensor.reshape(2, 2, 3)
    print(f"3D tensor: \n{three_dim}")
    permuted = three_dim.permute(2, 0, 1)  # swap dimensions
    print(f"Permuted: \n{permuted}")
    
    # 6. Device management (CPU/GPU)
    print("\n===== DEVICE MANAGEMENT =====")
    # Create a tensor on CPU
    cpu_tensor = torch.rand(3, 3)
    print(f"CPU tensor device: {cpu_tensor.device}")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        gpu_tensor = cpu_tensor.cuda()  # or cpu_tensor.to('cuda')
        print(f"GPU tensor device: {gpu_tensor.device}")
        
        # Move back to CPU
        back_to_cpu = gpu_tensor.cpu()  # or gpu_tensor.to('cpu')
        print(f"Back to CPU: {back_to_cpu.device}")
    else:
        print("CUDA not available. Running on CPU only.")
    
    # 7. Type conversions
    print("\n===== TYPE CONVERSIONS =====")
    float_tensor = torch.ones(2, 2)
    print(f"Float tensor: {float_tensor.dtype}")
    
    # Convert to different types
    int_tensor = float_tensor.int()
    print(f"Int tensor: {int_tensor.dtype}")
    
    double_tensor = float_tensor.double()  # or float_tensor.to(torch.float64)
    print(f"Double tensor: {double_tensor.dtype}")
    
    # 8. In-place operations (denoted by trailing underscore)
    print("\n===== IN-PLACE OPERATIONS =====")
    x = torch.ones(2, 2)
    print(f"Original x: \n{x}")
    
    x.add_(5)  # In-place addition (x = x + 5)
    print(f"After in-place addition: \n{x}")
    
    # Warning: In-place operations can cause issues with autograd
    print("\nNote: Be careful with in-place operations when using autograd!")
    
    return "Tensor basics tutorial completed!"

# Run the tutorial
if __name__ == "__main__":
    tensor_basics_tutorial()