import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SimpleNN(nn.Module):
    """
    A simple neural network with two hidden layers.
    
    Architecture:
    - Input layer: input_size neurons
    - Hidden layer 1: hidden_size neurons with ReLU activation
    - Hidden layer 2: hidden_size//2 neurons with ReLU activation
    - Output layer: output_size neurons
    
    Time Complexity: O(n*m) where n is batch size and m is total parameters
    Memory Complexity: O(n*m) for storing activations and gradients
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        """
        Initialize the neural network layers.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in the first hidden layer
            output_size (int): Number of output classes/values
            dropout_rate (float): Probability of zeroing elements during dropout
        """
        super(SimpleNN, self).__init__()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Batch normalization for better training stability
        self.bn1 = nn.BatchNorm1d(hidden_size)
        # Dropout to prevent overfitting
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second hidden layer (with reduced size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights using He initialization
        # Improves convergence for ReLU networks
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)  # Xavier for output layer
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_size]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_size]
        """
        # First hidden layer with ReLU activation
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second hidden layer with ReLU activation
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer (no activation - will be applied in loss function)
        x = self.fc3(x)
        return x

# A more complex CNN model for image tasks
class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.
    
    Architecture:
    - Conv layers with batch normalization and max pooling
    - Fully connected layers with dropout
    
    Time Complexity: O(n*c*h*w) where n is batch size, c is channels, h,w are dimensions
    Memory Complexity: O(n*c*h*w) for storing activations and gradients
    """
    def __init__(self, num_classes=10):
        """
        Initialize the CNN model.
        
        Args:
            num_classes (int): Number of output classes
        """
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Calculate the size after convolutions and pooling
        # For a 28x28 input (like MNIST), after 3 pooling layers: 28/(2^3) = 3.5 → 3
        # We need to account for this in the fully connected layer
        self.fc_input_size = 128 * 3 * 3
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc_bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        # Apply first convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Apply second convolutional block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Apply third convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        # -1 means infer this dimension from the other dimensions
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = F.relu(self.fc_bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def neural_networks_tutorial():
    """
    Demonstrates building and training neural networks with PyTorch.
    """
    # Import modules inside the function to ensure they're accessible
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    
    print("===== NEURAL NETWORKS WITH PYTORCH =====")
    
    # 1. Create a simple neural network
    input_size = 10
    hidden_size = 50
    output_size = 3
    
    model = SimpleNN(input_size, hidden_size, output_size)
    print(f"Model architecture:\n{model}")
    
    # Count the number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params}")
    
    # 2. Create a sample input tensor
    x = torch.randn(8, input_size)  # Batch size of 8
    
    # 3. Forward pass
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 4. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print(f"\nLoss function: {criterion}")
    print(f"Optimizer: {optimizer}")
    
    # 5. Generate synthetic data for training demonstration
    print("\n===== TRAINING DEMONSTRATION WITH SYNTHETIC DATA =====")
    
    # Create random data - 100 samples, 10 features
    X_train = torch.randn(100, input_size)
    
    # Create random target classes (0, 1, or 2)
    y_train = torch.randint(0, output_size, (100,))
    
    # Create DataLoader for batch processing
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 6. Training loop
    num_epochs = 5
    train_losses = []
    
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            running_loss += loss.item()
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # 7. Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # 8. Evaluation mode and inference
    print("\n===== EVALUATION AND INFERENCE =====")
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate test data
    X_test = torch.randn(10, input_size)
    
    # Make predictions (no gradient tracking needed)
    with torch.no_grad():
        predictions = model(X_test)
        # Get the predicted class (index of maximum value)
        _, predicted_classes = torch.max(predictions, 1)
    
    print(f"Test input shape: {X_test.shape}")
    print(f"Raw predictions shape: {predictions.shape}")
    print(f"Predicted classes: {predicted_classes}")
    
    # 9. Saving and loading models
    print("\n===== SAVING AND LOADING MODELS =====")
    
    # Save model weights
    torch.save(model.state_dict(), 'simple_nn_weights.pth')
    print("Model weights saved to 'simple_nn_weights.pth'")
    
    # Save entire model
    torch.save(model, 'simple_nn_model.pth')
    print("Complete model saved to 'simple_nn_model.pth'")
    
    # Load model weights (to a new model)
    new_model = SimpleNN(input_size, hidden_size, output_size)
    new_model.load_state_dict(torch.load('simple_nn_weights.pth'))
    print("\nLoaded weights into a new model")
    
    # FIX for PyTorch 2.6+ - METHOD 1: Using weights_only=False parameter
    # This method is less secure but simpler
    print("\n--- METHOD 1: Using weights_only=False parameter ---")
    loaded_model = torch.load('simple_nn_model.pth', weights_only=False)
    print("Loaded the complete model with weights_only=False")
    
    # FIX for PyTorch 2.6+ - METHOD 2: Using safe_globals context manager
    # This method is more secure and recommended for production
    print("\n--- METHOD 2: Using safe_globals context manager (recommended) ---")
    import torch.serialization
    
    # We need to include all classes used in the model
    # This includes PyTorch internal classes like Linear, BatchNorm, etc.
    from torch.nn.modules.linear import Linear
    from torch.nn.modules.batchnorm import BatchNorm1d
    from torch.nn.modules.dropout import Dropout
    from torch.nn.parameter import Parameter
    
    # Create a comprehensive list of classes that might be in the pickled model
    safe_classes = [
        SimpleNN,                # Our custom model class
        Linear, BatchNorm1d, Dropout,  # Layer types used in our model
        Parameter,               # For model parameters
        dict, list, tuple, set,  # Common container types
        torch.Tensor,            # Tensor objects
        type(None)               # None values
    ]
    
    # Now use safe_globals with the complete list
    with torch.serialization.safe_globals(safe_classes):
        loaded_model = torch.load('simple_nn_model.pth')
        print("Loaded the complete model using safe_globals context manager")
    
    # METHOD 3 (Simplest & Best Practice): Always save and load state_dict instead of entire model
    print("\n--- METHOD 3: Best Practice for New Projects ---")
    print("For new projects, it's best to save just the state_dict rather than the entire model:")
    print("torch.save(model.state_dict(), 'model_weights.pth')  # Save just weights")
    print("new_model.load_state_dict(torch.load('model_weights.pth'))  # Load weights to a new model")
    print("This approach avoids all pickle-related security issues")
    # Summary of model loading approaches
    print("\n===== SUMMARY OF MODEL LOADING APPROACHES =====")
    print("1. weights_only=False: Quick fix, but less secure - use only with trusted files")
    print("2. safe_globals: More secure but complex - requires listing all classes")
    print("3. state_dict only: Best practice - save and load only the weights (no pickle security issues)")
    print("For new projects, prefer option 3 - save and load just state_dict instead of entire models")
    
    # 10. Using GPU if available
    print("\n===== USING GPU ACCELERATION =====")
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to GPU
    model.to(device)
    print(f"Model moved to {device}")
    
    # When using GPU, remember to move input data to the same device
    if torch.cuda.is_available():
        x_gpu = X_test.to(device)
        output_gpu = model(x_gpu)
        print(f"GPU output shape: {output_gpu.shape}")
    
    # 11. Alternative Neural Network Architectures
    print("\n===== ALTERNATIVE NEURAL NETWORK ARCHITECTURES =====")
    
    # Create a CNN model
    cnn_model = SimpleCNN(num_classes=10)
    print(f"CNN Architecture:\n{cnn_model}")
    
    # 12. PyTorch Built-in Models
    print("\n===== USING PYTORCH BUILT-IN MODELS =====")
    
    # PyTorch has many pre-built models in torchvision
    print("PyTorch offers pre-trained models like:")
    print("- ResNet (torchvision.models.resnet18)")
    print("- VGG (torchvision.models.vgg16)")
    print("- DenseNet (torchvision.models.densenet121)")
    print("- EfficientNet (torchvision.models.efficientnet_b0)")
    
    # 13. Tips for Neural Network Implementation
    print("\n===== TIPS FOR NEURAL NETWORK IMPLEMENTATION =====")
    print("1. Start with simple architectures and gradually increase complexity")
    print("2. Use batch normalization to stabilize training")
    print("3. Apply dropout to prevent overfitting")
    print("4. Experiment with different optimizers (Adam, SGD with momentum)")
    print("5. Implement learning rate scheduling for better convergence")
    print("6. Use proper weight initialization")
    print("7. Apply early stopping to prevent overfitting")
    print("8. Monitor both training and validation loss")
    
    return "Neural networks tutorial completed!"

# This would run the tutorial if executed directly
if __name__ == "__main__":
    neural_networks_tutorial()