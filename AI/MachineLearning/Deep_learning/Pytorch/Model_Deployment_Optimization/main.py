import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub, prepare, convert, QConfig
import torchvision.transforms as transforms

class SimpleCNN(nn.Module):
    """
    A simple CNN for demonstration of optimization techniques.
    
    Architecture:
    - 2 convolutional layers with ReLU activations and batch normalization
    - 2 fully connected layers
    
    Time Complexity: O(batch_size * channels * height * width)
    Memory Complexity: O(batch_size * filters * feature_map_size)
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Assuming input image size is 32x32
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # For quantization-aware training
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def forward(self, x):
        # Quantize input if training with quantization
        x = self.quant(x)
        
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten for fully connected layers
        # Make tensor contiguous before reshaping and use reshape instead of view
        x = x.contiguous().reshape(-1, 32 * 8 * 8)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Dequantize output
        x = self.dequant(x)
        
        return x

def export_to_onnx(model, input_shape, filename="model.onnx"):
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, channels, height, width)
        filename: Output filename
    """
    # Create a dummy input tensor of the specified shape
    dummy_input = torch.randn(input_shape)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Export the model
    torch.onnx.export(
        model,                      # model being run
        dummy_input,                # model input (or a tuple for multiple inputs)
        filename,                   # where to save the model
        export_params=True,         # store the trained parameter weights inside the model file
        opset_version=12,           # the ONNX version to export the model to
        do_constant_folding=True,   # optimization: fold constants
        input_names=['input'],      # the model's input names
        output_names=['output'],    # the model's output names
        dynamic_axes={              # support for variable length axes
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {filename}")

def export_to_torchscript(model, input_shape, filename="model.pt", scripting=True):
    """
    Export a PyTorch model to TorchScript format.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor
        filename: Output filename
        scripting: Whether to use scripting (True) or tracing (False)
    """
    # Set the model to evaluation mode
    model.eval()
    
    if scripting:
        # Use scripting
        scripted_model = torch.jit.script(model)
        scripted_model.save(filename)
        print(f"Scripted model saved to {filename}")
        return scripted_model
    else:
        # Use tracing with a dummy input
        dummy_input = torch.randn(input_shape)
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(filename)
        print(f"Traced model saved to {filename}")
        return traced_model

def export_to_mobile(model, input_shape, filename="model_mobile.pt"):
    """
    Export a PyTorch model optimized for mobile deployment.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor
        filename: Output filename
    """
    # Set the model to evaluation mode
    model.eval()
    
    # First, convert to TorchScript via tracing
    dummy_input = torch.randn(input_shape)
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Optimize for mobile
    optimized_model = optimize_for_mobile(traced_model)
    
    # Save the optimized model
    optimized_model._save_for_lite_interpreter(filename)
    print(f"Model optimized for mobile saved to {filename}")
    return optimized_model

def quantize_model(model, calibration_loader=None, static=False):
    """
    Quantize a PyTorch model to reduce its size and improve inference speed.
    
    Args:
        model: PyTorch model
        calibration_loader: DataLoader for calibration data (only needed for static quantization)
        static: Whether to use static (True) or dynamic (False) quantization
        
    Returns:
        Quantized model
    """
    # Set the model to evaluation mode
    model.eval()
    
    if static and calibration_loader is not None:
        # Static quantization (post-training)
        # Configure qconfig
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for static quantization
        model_prepared = prepare(model)
        
        # Calibrate with the training set
        print("Calibrating model for static quantization...")
        with torch.no_grad():
            for inputs, _ in calibration_loader:
                model_prepared(inputs)
        
        # Convert to quantized model
        quantized_model = convert(model_prepared)
        print("Static quantization completed")
    else:
        # Dynamic quantization
        quantized_model = quantize_dynamic(
            model, 
            {nn.Linear, nn.LSTM, nn.GRU, nn.LSTMCell, nn.RNNCell, nn.GRUCell},  # Quantize all linear and RNN layers
            dtype=torch.qint8
        )
        print("Dynamic quantization completed")
    
    return quantized_model

def prune_model(model, pruning_method="l1_unstructured", amount=0.3):
    """
    Apply pruning to a PyTorch model to reduce its size.
    
    Args:
        model: PyTorch model
        pruning_method: Method of pruning ('l1_unstructured', 'random_unstructured', etc.)
        amount: Amount of weights to prune (between 0 and 1)
        
    Returns:
        Pruned model
    """
    try:
        import torch.nn.utils.prune as prune
        
        # Map of pruning methods
        pruning_methods = {
            "l1_unstructured": prune.l1_unstructured,
            "random_unstructured": prune.random_unstructured,
            "ln_structured": prune.ln_structured
        }
        
        # Get the specified pruning method
        if pruning_method not in pruning_methods:
            print(f"Unknown pruning method: {pruning_method}. Falling back to l1_unstructured.")
            pruning_method = "l1_unstructured"
        
        pruning_fn = pruning_methods[pruning_method]
        
        # Apply pruning to all convolutional and linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                pruning_fn(module, name='weight', amount=amount)
        
        # Make pruning permanent
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.remove(module, 'weight')
        
        print(f"Model pruned using {pruning_method} method with amount={amount}")
        return model
    
    except ImportError:
        print("Pruning requires PyTorch 1.4 or newer.")
        return model

def benchmark_inference(model, input_shape, num_runs=100):
    """
    Benchmark the inference time of a model.
    
    Args:
        model: PyTorch model or TorchScript model
        input_shape: Shape of input tensor
        num_runs: Number of inference runs for timing
        
    Returns:
        tuple: (avg_time_ms, model_size_mb)
    """
    # Check if the model is a TorchScript model
    is_torchscript = isinstance(model, torch.jit.ScriptModule) or hasattr(model, '_c')
    
    # Set the model to evaluation mode if it's a regular PyTorch model
    if not is_torchscript and hasattr(model, 'eval'):
        model.eval()
    
    # Create a dummy input
    dummy_input = torch.randn(input_shape)
    
    # Move everything to the same device as the model
    if is_torchscript:
        # For TorchScript models (including mobile), use CPU
        device = torch.device('cpu')
    else:
        # For regular PyTorch models, get device from parameters
        device = next(model.parameters()).device
    
    dummy_input = dummy_input.to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Time inference
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(dummy_input)
        end_time = time.time()
    
    # Calculate average inference time
    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    
    # Calculate model size
    if is_torchscript:
        # For TorchScript models, check file size
        # Note: This will only work if the model has been saved to disk
        try:
            if hasattr(model, '_save_to_file'):
                # For mobile-optimized models, we need a temporary file
                temp_filename = "temp_model_for_size.pt"
                if "_save_for_lite_interpreter" in dir(model):
                    model._save_for_lite_interpreter(temp_filename)
                else:
                    model._save_to_file(temp_filename)
                model_size_mb = os.path.getsize(temp_filename) / (1024 * 1024)
                # Clean up temporary file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            else:
                # If we can't save, estimate size based on serialization
                buffer = io.BytesIO()
                torch.jit.save(model, buffer)
                model_size_mb = len(buffer.getvalue()) / (1024 * 1024)
        except Exception as e:
            print(f"Warning: Failed to get model size, using estimate: {e}")
            # Rough estimate for TorchScript models
            model_size_mb = 0.01  # Arbitrary small size as placeholder
    else:
        # Regular PyTorch model
        model_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    return avg_time_ms, model_size_mb

def compare_optimization_techniques():
    """
    Compare different model optimization techniques.
    
    Returns:
        tuple: (inference_times, model_sizes)
    """
    # Create a model for benchmarking
    model = SimpleCNN(num_classes=10)
    
    # Input shape: (batch_size, channels, height, width)
    input_shape = (1, 3, 32, 32)
    
    # Create a simple calibration dataset for quantization
    dummy_input = torch.randn(20, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (20,))
    dummy_dataset = TensorDataset(dummy_input, dummy_target)
    calibration_loader = DataLoader(dummy_dataset, batch_size=4)
    
    # Results containers
    methods = ['Original', 'TorchScript', 'Quantized (Dynamic)', 'Quantized (Static)', 'Pruned', 'Mobile Optimized']
    inference_times = []
    model_sizes = []
    
    # 1. Benchmark original model
    print("\n===== BENCHMARKING ORIGINAL MODEL =====")
    model.eval()
    avg_time, model_size = benchmark_inference(model, input_shape)
    inference_times.append(avg_time)
    model_sizes.append(model_size)
    print(f"Original model - Avg inference time: {avg_time:.2f} ms, Size: {model_size:.2f} MB")
    
    # 2. Benchmark TorchScript model
    print("\n===== BENCHMARKING TORCHSCRIPT MODEL =====")
    scripted_model = export_to_torchscript(model, input_shape, "demo_model.pt")
    avg_time, model_size = benchmark_inference(scripted_model, input_shape)
    inference_times.append(avg_time)
    model_sizes.append(model_size)
    print(f"TorchScript model - Avg inference time: {avg_time:.2f} ms, Size: {model_size:.2f} MB")
    
    # 3. Benchmark dynamically quantized model
    print("\n===== BENCHMARKING DYNAMICALLY QUANTIZED MODEL =====")
    quantized_dynamic = quantize_model(model, static=False)
    avg_time, model_size = benchmark_inference(quantized_dynamic, input_shape)
    inference_times.append(avg_time)
    model_sizes.append(model_size)
    print(f"Dynamically quantized model - Avg inference time: {avg_time:.2f} ms, Size: {model_size:.2f} MB")
    
    # 4. Benchmark statically quantized model
    print("\n===== BENCHMARKING STATICALLY QUANTIZED MODEL =====")
    quantized_static = quantize_model(model, calibration_loader, static=True)
    avg_time, model_size = benchmark_inference(quantized_static, input_shape)
    inference_times.append(avg_time)
    model_sizes.append(model_size)
    print(f"Statically quantized model - Avg inference time: {avg_time:.2f} ms, Size: {model_size:.2f} MB")
    
    # 5. Benchmark pruned model
    print("\n===== BENCHMARKING PRUNED MODEL =====")
    pruned_model = prune_model(model.cpu(), amount=0.3)  # Prune 30% of weights
    avg_time, model_size = benchmark_inference(pruned_model, input_shape)
    inference_times.append(avg_time)
    model_sizes.append(model_size)
    print(f"Pruned model - Avg inference time: {avg_time:.2f} ms, Size: {model_size:.2f} MB")
    
    # 6. Benchmark mobile optimized model
    print("\n===== BENCHMARKING MOBILE OPTIMIZED MODEL =====")
    mobile_model = export_to_mobile(model, input_shape, "demo_model_mobile.pt")
    avg_time, model_size = benchmark_inference(mobile_model, input_shape)
    inference_times.append(avg_time)
    model_sizes.append(model_size)
    print(f"Mobile optimized model - Avg inference time: {avg_time:.2f} ms, Size: {model_size:.2f} MB")
    
    return methods, inference_times, model_sizes

def model_deployment_tutorial():
    """
    Demonstrates model deployment and optimization techniques with PyTorch.
    """
    print("===== MODEL DEPLOYMENT AND OPTIMIZATION WITH PYTORCH =====")
    
    # 1. Overview of deployment considerations
    print("\n===== DEPLOYMENT CONSIDERATIONS =====")
    print("1. Model Size: Affects storage requirements and loading time")
    print("2. Inference Speed: Critical for real-time applications")
    print("3. Hardware Constraints: Edge devices, mobile, cloud, etc.")
    print("4. Batch Processing: Trade-off between latency and throughput")
    print("5. Scalability: Handling multiple concurrent requests")
    
    # 2. Model Export Formats
    print("\n===== MODEL EXPORT FORMATS =====")
    
    print("TorchScript:")
    print("- Serialized and optimized PyTorch models")
    print("- Two methods: Tracing and Scripting")
    print("- Good for production deployment in C++ applications")
    
    print("\nONNX (Open Neural Network Exchange):")
    print("- Industry standard for neural network interchange")
    print("- Enables deployment across different frameworks")
    print("- Supports hardware-specific optimizations")
    
    print("\nTorchServe:")
    print("- PyTorch's model serving library")
    print("- Provides HTTP endpoints for model inference")
    print("- Supports model versioning and A/B testing")
    
    print("\nMobile Deployment:")
    print("- PyTorch Mobile for iOS and Android")
    print("- Optimized for on-device inference")
    print("- Reduced binary size and memory footprint")
    
    # 3. Optimization Techniques
    print("\n===== MODEL OPTIMIZATION TECHNIQUES =====")
    
    print("Quantization:")
    print("- Reduces precision of weights (e.g., FP32 to INT8)")
    print("- Types: Dynamic, Static, and Quantization-Aware Training")
    print("- Benefits: Smaller model size, faster inference, lower memory usage")
    
    print("\nPruning:")
    print("- Removes less important weights from the model")
    print("- Types: Unstructured, Structured, and Magnitude-based")
    print("- Benefits: Reduced model size and potentially faster inference")
    
    print("\nKnowledge Distillation:")
    print("- Trains a smaller 'student' model to mimic a larger 'teacher' model")
    print("- Benefits: Compact models with performance close to larger models")
    
    print("\nFusion of Operations:")
    print("- Combines multiple operations into a single optimized operation")
    print("- Benefits: Reduced memory overhead and improved execution speed")
    
    # 4. Deployment Platforms
    print("\n===== DEPLOYMENT PLATFORMS =====")
    
    print("Cloud Deployment:")
    print("- AWS, Azure, Google Cloud")
    print("- Container orchestration (Kubernetes, Docker)")
    print("- Auto-scaling and high availability")
    
    print("\nEdge Deployment:")
    print("- NVIDIA Jetson, Intel NCS, Google Edge TPU")
    print("- Reduced latency and bandwidth usage")
    print("- Often requires model optimization")
    
    print("\nMobile Deployment:")
    print("- iOS (CoreML integration)")
    print("- Android (TensorFlow Lite integration or PyTorch Mobile)")
    print("- Requires significant model optimization")
    
    print("\nWeb Deployment:")
    print("- TensorFlow.js or ONNX.js")
    print("- Client-side inference in browsers")
    print("- Limited by browser capabilities and client hardware")
    
    # 5. Create and export a simple model
    print("\n===== CREATING AND EXPORTING A MODEL =====")
    
    # Create a simple model
    model = SimpleCNN(num_classes=10)
    print(f"Created a simple CNN model for demonstration")
    
    # Define input shape (batch_size, channels, height, width)
    input_shape = (1, 3, 32, 32)
    
    # Export to ONNX
    export_to_onnx(model, input_shape, "demo_model.onnx")
    
    # Export to TorchScript
    export_to_torchscript(model, input_shape, "demo_model_script.pt", scripting=True)
    export_to_torchscript(model, input_shape, "demo_model_trace.pt", scripting=False)
    
    # Export for mobile
    export_to_mobile(model, input_shape, "demo_model_mobile.pt")
    
    # 6. Compare optimization techniques
    print("\n===== COMPARING OPTIMIZATION TECHNIQUES =====")
    methods, inference_times, model_sizes = compare_optimization_techniques()
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot inference times
    ax1.bar(methods, inference_times, color='skyblue')
    ax1.set_ylabel('Average Inference Time (ms)')
    ax1.set_title('Inference Time Comparison')
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    
    # Plot model sizes
    ax2.bar(methods, model_sizes, color='salmon')
    ax2.set_ylabel('Model Size (MB)')
    ax2.set_title('Model Size Comparison')
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # 7. Best practices
    print("\n===== BEST PRACTICES FOR MODEL DEPLOYMENT =====")
    
    print("1. Benchmark before and after optimization")
    print("2. Consider trade-offs between size, speed, and accuracy")
    print("3. Match optimization techniques to deployment target")
    print("4. Test deployment in the target environment")
    print("5. Monitor model performance in production")
    print("6. Implement A/B testing for model updates")
    print("7. Design for scalability and reliability")
    print("8. Include versioning and rollback capabilities")
    print("9. Consider model update strategies")
    print("10. Implement proper error handling and logging")
    
    # 8. Additional resources
    print("\n===== ADDITIONAL RESOURCES =====")
    print("- PyTorch Documentation: https://pytorch.org/docs/stable/index.html")
    print("- PyTorch Mobile: https://pytorch.org/mobile/home/")
    print("- ONNX: https://onnx.ai/")
    print("- TorchServe: https://pytorch.org/serve/")
    print("- PyTorch Blog for case studies: https://pytorch.org/blog/")
    
    return "Model deployment tutorial completed!"

# Run the tutorial if this file is executed directly
if __name__ == "__main__":
    model_deployment_tutorial()