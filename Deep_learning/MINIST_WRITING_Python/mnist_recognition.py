import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from PIL import Image
from torchvision import transforms

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define paths
TRAIN_IMAGES_PATH = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-images.idx3-ubyte"
TRAIN_LABELS_PATH = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-labels.idx1-ubyte"
TEST_IMAGES_PATH = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-images.idx3-ubyte"
TEST_LABELS_PATH = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-labels.idx1-ubyte"

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------- Data Loading Functions ---------------------

def read_idx_images(file_path):
    """
    Read IDX image file format used by MNIST.
    
    Args:
        file_path: Path to the IDX file
        
    Returns:
        numpy array of images with shape (num_images, height, width)
    """
    try:
        with open(file_path, 'rb') as f:
            # Read header information
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            
            # Verify magic number for images (2051)
            if magic != 2051:
                raise ValueError(f"Invalid magic number {magic} in {file_path}")
            
            # Read image data
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
            image_data = image_data.reshape(num_images, rows, cols)
            
            print(f"Loaded {num_images} images with shape ({rows}, {cols})")
            return image_data
    except Exception as e:
        print(f"Error reading image file {file_path}: {e}")
        raise

def read_idx_labels(file_path):
    """
    Read IDX label file format used by MNIST.
    
    Args:
        file_path: Path to the IDX file
        
    Returns:
        numpy array of labels
    """
    try:
        with open(file_path, 'rb') as f:
            # Read header information
            magic, num_labels = struct.unpack('>II', f.read(8))
            
            # Verify magic number for labels (2049)
            if magic != 2049:
                raise ValueError(f"Invalid magic number {magic} in {file_path}")
            
            # Read label data
            label_data = np.frombuffer(f.read(), dtype=np.uint8)
            
            print(f"Loaded {num_labels} labels")
            return label_data
    except Exception as e:
        print(f"Error reading label file {file_path}: {e}")
        raise

# --------------------- Custom Dataset Class with Augmentation ---------------------

class MNISTDataset(Dataset):
    """
    Custom Dataset for MNIST with optional data augmentation
    """
    def __init__(self, images, labels, transform=None, augment=False):
        """
        Initialize the dataset with images and labels.
        
        Args:
            images: numpy array of images
            labels: numpy array of labels
            transform: optional transform to be applied to the images
            augment: whether to apply data augmentation
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
        # Define augmentation transforms
        self.augmentation = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            transforms.ColorJitter(brightness=0.2),
        ])
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get an item by index"""
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensors
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Apply data augmentation during training if enabled
        if self.augment:
            # Convert to PIL Image for transformation
            image_pil = transforms.ToPILImage()(image_tensor)
            # Apply augmentation
            image_pil = self.augmentation(image_pil)
            # Convert back to tensor
            image_tensor = transforms.ToTensor()(image_pil)
            
        # Apply custom transforms if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, label_tensor

# --------------------- Improved Neural Network Model ---------------------

class ImprovedMNISTNet(nn.Module):
    """
    Enhanced Neural Network for MNIST digit recognition with deeper architecture
    and regularization techniques for better accuracy
    """
    def __init__(self, dropout_rate=0.4):
        """Initialize the network layers with improved architecture"""
        super(ImprovedMNISTNet, self).__init__()
        
        # First Convolutional Block
        # Input: 1x28x28, Output: 32x28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second Convolutional Block
        # Input: 32x14x14 (after pooling), Output: 64x14x14
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third Convolutional Block
        # Input: 64x7x7 (after pooling), Output: 128x7x7
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fourth Convolutional Block for more feature extraction
        # Input: 128x7x7, Output: 256x7x7
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Calculate the size of flattened features
        # After 2 max pooling layers (2x2), the 28x28 image becomes 7x7
        flattened_size = 256 * 7 * 7
        
        # Fully connected layers with more capacity
        self.fc1 = nn.Linear(flattened_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)
    
    def forward(self, x):
        """Forward pass through the network with residual-like connections"""
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)  # Using LeakyReLU for better gradient flow
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # Second conv block
        x_res = x  # Save for residual connection
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        # Add residual-like connection by upsampling the previous feature map
        if x_res.size(1) != x.size(1):  # If channel dimensions don't match
            x_res = F.pad(x_res, (0, 0, 0, 0, 0, x.size(1) - x_res.size(1)))
        x = x + x_res  # Residual connection
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        
        # Third conv block
        x_res = x  # Save for residual connection
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        # Add residual-like connection
        if x_res.size(1) != x.size(1):
            x_res = F.pad(x_res, (0, 0, 0, 0, 0, x.size(1) - x_res.size(1)))
        x = x + x_res  # Residual connection
        
        # Fourth conv block
        x_res = x  # Save for residual connection
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        # Add residual-like connection
        if x_res.size(1) != x.size(1):
            x_res = F.pad(x_res, (0, 0, 0, 0, 0, x.size(1) - x_res.size(1)))
        x = x + x_res  # Residual connection
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
        
    def predict(self, x):
        """Make a prediction by taking the class with highest probability"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities

# --------------------- Training Functions ---------------------

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """
    Train the model for one epoch with enhanced monitoring.
    
    Args:
        model: The neural network model
        dataloader: DataLoader providing the training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to train on (CPU or GPU)
        scheduler: Learning rate scheduler
        
    Returns:
        Average loss for the epoch and accuracy
    """
    model.train()  # Set model to training mode
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move data to the appropriate device
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Step the OneCycleLR scheduler after each batch if it's being used
        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # Track statistics
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch: {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')
    
    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total
    
    return epoch_loss, epoch_accuracy

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation/test set.
    
    Args:
        model: The neural network model
        dataloader: DataLoader providing the validation/test data
        criterion: Loss function
        device: Device to evaluate on (CPU or GPU)
        
    Returns:
        Average loss and accuracy for the dataset
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Confusion matrix to track per-class performance
    confusion_matrix = torch.zeros(10, 10, dtype=torch.long)
    
    with torch.no_grad():  # No need to track gradients during evaluation
        for images, labels in dataloader:
            # Move data to the appropriate device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track statistics
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i, acc in enumerate(per_class_accuracy):
        print(f"Digit {i}: {acc.item()*100:.2f}%")
    
    return avg_loss, accuracy

def save_model(model, path="improved_mnist_model.pth"):
    """
    Save the trained model to disk.
    
    Args:
        model: The trained model
        path: Path to save the model to
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path="improved_mnist_model.pth", device=None):
    """
    Load a trained model from disk.
    
    Args:
        model_class: The class of the model to load
        path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        The loaded model
    """
    model = model_class()
    if device:
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        model.load_state_dict(torch.load(path))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {path}")
    return model

# --------------------- Visualization Functions ---------------------

def visualize_results(images, labels, predictions, num_samples=10):
    """
    Visualize some test results.
    
    Args:
        images: Tensor of images
        labels: Tensor of ground truth labels
        predictions: Tensor of predicted labels
        num_samples: Number of samples to visualize
    """
    # Convert tensors to numpy arrays
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Find some misclassified examples
    misclassified_idxs = np.where(predictions != labels)[0]
    
    # Visualize a mix of correct and incorrect predictions
    correct_shown = 0
    incorrect_shown = 0
    correct_target = num_samples // 2
    incorrect_target = num_samples - correct_target
    
    for i in range(len(labels)):
        if predictions[i] == labels[i] and correct_shown < correct_target:
            idx = i
            correct_shown += 1
            is_correct = True
        elif predictions[i] != labels[i] and incorrect_shown < incorrect_target:
            idx = i
            incorrect_shown += 1
            is_correct = False
        else:
            continue
        
        # Get index for subplot
        plot_idx = correct_shown - 1 if is_correct else correct_target + incorrect_shown - 1
        if plot_idx >= num_samples:
            break
            
        # Plot the image
        axes[plot_idx].imshow(images[idx, 0], cmap='gray')
        color = 'green' if is_correct else 'red'
        axes[plot_idx].set_title(f'True: {labels[idx]}, Pred: {predictions[idx]}', color=color)
        axes[plot_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_results.png')
    plt.show()

def visualize_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    Visualize the training history.
    
    Args:
        train_losses: List of training losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_losses: List of validation losses per epoch
        val_accuracies: List of validation accuracies per epoch
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot the losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot the accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('mnist_training_history.png')
    plt.show()

def visualize_feature_maps(model, image_tensor, device, layer_index=1):
    """
    Visualize feature maps of the model to understand what it's learning.
    
    Args:
        model: The neural network model
        image_tensor: Input image tensor
        device: Device to evaluate on (CPU or GPU)
        layer_index: Index of the convolutional layer to visualize (1, 2, 3, or 4)
    """
    model.eval()
    
    # Move image to device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Define hooks to capture feature maps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hooks for the desired layer
    if layer_index == 1:
        handle = model.conv1.register_forward_hook(get_activation('conv1'))
    elif layer_index == 2:
        handle = model.conv2.register_forward_hook(get_activation('conv2'))
    elif layer_index == 3:
        handle = model.conv3.register_forward_hook(get_activation('conv3'))
    elif layer_index == 4:
        handle = model.conv4.register_forward_hook(get_activation('conv4'))
    else:
        print(f"Invalid layer index: {layer_index}")
        return
    
    # Forward pass
    with torch.no_grad():
        model(image_tensor)
    
    # Remove the hook
    handle.remove()
    
    # Get feature maps
    layer_name = f'conv{layer_index}'
    feature_maps = activation[layer_name].squeeze(0).cpu()
    
    # Plot feature maps
    num_maps = min(16, feature_maps.size(0))
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_maps):
        axes[i].imshow(feature_maps[i], cmap='viridis')
        axes[i].axis('off')
    
    plt.suptitle(f'Feature Maps of {layer_name} Layer')
    plt.tight_layout()
    plt.savefig(f'feature_maps_layer{layer_index}.png')
    plt.show()

# --------------------- Main Training Function ---------------------

def train_mnist_model(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs, device, scheduler=None):
    """
    Train the MNIST model with improved training protocol.
    
    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        model: The neural network model
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Number of epochs to train for
        device: Device to train on (CPU or GPU)
        scheduler: Learning rate scheduler
        
    Returns:
        Trained model and training history
    """
    # Move model to the device
    model = model.to(device)
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Best validation accuracy to track improvement
    best_val_accuracy = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train for one epoch
        train_loss, train_accuracy = train_epoch(model, train_dataloader, criterion, optimizer, device, scheduler)
        
        # Evaluate on validation set
        val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)
        
        # Step the learning rate scheduler if it's not OneCycleLR
        # (OneCycleLR is stepped in the training loop)
        if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_accuracy)
            else:
                scheduler.step()
        
        # Record history
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print epoch results
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # Save the model if it's the best so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, "best_mnist_model.pth")
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
    
    # Visualize training history
    visualize_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
    
    # Load the best model for final evaluation
    model = load_model(model.__class__, "best_mnist_model.pth", device)
    
    return model, (train_losses, train_accuracies, val_losses, val_accuracies)

# --------------------- Inference and Image Processing ---------------------

def preprocess_external_image(image_path):
    """
    Preprocess an external image for inference with multiple preprocessing strategies
    and allow the user to select the best one.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of processed image tensors ready for the model
    """
    try:
        # Load image
        original_image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Display original image
        plt.figure(figsize=(5, 5))
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        plt.savefig(f'original_{os.path.basename(image_path)}.png')
        plt.show()
        
        # Create multiple preprocessing variants
        preprocessed_images = []
        titles = []
        
        # Strategy 1: Simple resize to 28x28 (minimal preprocessing)
        image1 = original_image.resize((28, 28), Image.LANCZOS)
        image_array1 = np.array(image1).astype(np.float32) / 255.0
        # Make sure digits are black (0) and background is white (1) - MNIST standard
        if np.mean(image_array1) < 0.5:
            image_array1 = 1.0 - image_array1
        preprocessed_images.append(image_array1)
        titles.append("Simple Resize")
        
        # Strategy 2: Center the digit with bounding box detection
        image2 = original_image.copy()
        image_array2 = np.array(image2)
        
        # Otsu's thresholding for more robust binarization
        from skimage.filters import threshold_otsu
        try:
            thresh = threshold_otsu(image_array2)
            binary_image = image_array2 < thresh  # Assuming digit is darker than background
        except:
            # Fallback to simple thresholding if Otsu's fails
            binary_image = image_array2 < 128
        
        # Find the bounding box of the digit
        rows = np.any(binary_image, axis=1)
        cols = np.any(binary_image, axis=0)
        
        if np.any(rows) and np.any(cols):
            # Get the non-zero areas (the digit)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Get aspect ratio to maintain shape
            width = x_max - x_min
            height = y_max - y_min
            
            # Add some padding
            padding = max(width, height) // 4
            y_min = max(0, y_min - padding)
            y_max = min(image_array2.shape[0] - 1, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(image_array2.shape[1] - 1, x_max + padding)
            
            # Crop to the digit with a bit of padding
            image2 = image2.crop((x_min, y_min, x_max, y_max))
            
            # Resize to 20x20 and pad to 28x28 to match MNIST format
            # Create a new square image to ensure aspect ratio is maintained
            size = max(image2.width, image2.height)
            square_image = Image.new('L', (size, size), 255)
            # Paste the digit in the center of the square
            paste_x = (size - image2.width) // 2
            paste_y = (size - image2.height) // 2
            square_image.paste(image2, (paste_x, paste_y))
            
            # Now resize to 20x20
            image2 = square_image.resize((20, 20), Image.LANCZOS)
            
            # Create a new white background image
            padded_image = Image.new('L', (28, 28), 255)
            # Paste the digit in the center
            padded_image.paste(image2, ((28 - 20) // 2, (28 - 20) // 2))
            image2 = padded_image
        else:
            # If no foreground is detected, just resize to 28x28
            image2 = image2.resize((28, 28), Image.LANCZOS)
        
        image_array2 = np.array(image2).astype(np.float32) / 255.0
        # Invert if necessary (MNIST has white background, black digits)
        if np.mean(image_array2) < 0.5:
            image_array2 = 1.0 - image_array2
        preprocessed_images.append(image_array2)
        titles.append("Centered with Padding")
        
        # Strategy 3: Adaptive thresholding for better noise handling
        image3 = original_image.copy()
        image_array3 = np.array(image3)
        
        # Apply adaptive thresholding
        try:
            from skimage.filters import threshold_local
            block_size = max(11, min(image_array3.shape) // 10 * 2 + 1)  # Must be odd
            thresh = threshold_local(image_array3, block_size=block_size, method='gaussian')
            binary_image = image_array3 < thresh
        except:
            # Fallback to global thresholding
            binary_image = image_array3 < np.mean(image_array3)
        
        # Convert back to image
        binary_array = (binary_image.astype(np.uint8) * 255)
        binary_image = Image.fromarray(binary_array)
        
        # Resize to 28x28
        image3 = binary_image.resize((28, 28), Image.LANCZOS)
        image_array3 = np.array(image3).astype(np.float32) / 255.0
        
        # Invert if necessary (MNIST has white background, black digits)
        if np.mean(image_array3) < 0.5:
            image_array3 = 1.0 - image_array3
        preprocessed_images.append(image_array3)
        titles.append("Adaptive Threshold")
        
        # Strategy 4: Edge detection to focus on digit shape
        try:
            from skimage import feature
            # Use Canny edge detection
            image_array4 = np.array(original_image)
            edges = feature.canny(image_array4 / 255.0, sigma=1)
            edge_image = Image.fromarray((edges * 255).astype(np.uint8))
            
            # Resize to 28x28
            image4 = edge_image.resize((28, 28), Image.LANCZOS)
            image_array4 = np.array(image4).astype(np.float32) / 255.0
            
            # Invert if necessary (MNIST has white background, black digits)
            if np.mean(image_array4) < 0.5:
                image_array4 = 1.0 - image_array4
            preprocessed_images.append(image_array4)
            titles.append("Edge Detection")
        except:
            # Skip if edge detection fails
            pass
        
        # Display all preprocessing variants
        plt.figure(figsize=(15, 4))
        for i, (img_array, title) in enumerate(zip(preprocessed_images, titles)):
            plt.subplot(1, len(preprocessed_images), i+1)
            plt.imshow(img_array, cmap='gray')
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'preprocessing_variants_{os.path.basename(image_path)}.png')
        plt.show()
        
        # Convert all variants to tensors
        image_tensors = []
        for img_array in preprocessed_images:
            # Convert to tensor and add batch and channel dimensions
            tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            image_tensors.append(tensor)
        
        return image_tensors
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def infer_digit(model, image_path, device):
    """
    Predict the digit in an image with multiple preprocessing strategies.
    
    Args:
        model: Trained model
        image_path: Path to the image file
        device: Device to run inference on
        
    Returns:
        Predicted digit and confidence
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Load and preprocess the image with multiple strategies
    image_tensors = preprocess_external_image(image_path)
    
    if not image_tensors:
        return None, None
    
    # Make predictions for each preprocessing variant
    results = []
    
    for i, image_tensor in enumerate(image_tensors):
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            predicted_class, probabilities = model.predict(image_tensor)
        
        # Get probabilities as numpy array
        probs = probabilities.cpu().numpy()[0]
        
        # Get the confidence score (probability of the predicted class)
        confidence = probs[predicted_class.item()] * 100
        
        # Store the result
        results.append({
            'variant': i,
            'predicted_class': predicted_class.item(),
            'confidence': confidence,
            'probabilities': probs,
            'image_tensor': image_tensor
        })
    
    # Find the variant with highest confidence
    best_result = max(results, key=lambda x: x['confidence'])
    
    # Also use test-time augmentation on the best variant
    tta_digit, tta_probs = test_time_augmentation(
        model, best_result['image_tensor'], device, num_augmentations=5
    )
    
    # Visualize all predictions with a summary
    plt.figure(figsize=(15, 5 + 4 * (len(results) > 2)))
    
    # Define preprocessing strategy names
    strategy_names = ["Simple Resize", "Centered with Padding", "Adaptive Threshold", "Edge Detection"]
    
    # First plot: summary of all preprocessing strategies
    plt.subplot(1, 3, 1)
    
    # Create a horizontal bar chart of confidences
    variants = [strategy_names[r['variant']] for r in results]
    confidences = [r['confidence'] for r in results]
    predictions = [r['predicted_class'] for r in results]
    
    # Sort by confidence
    sorted_indices = np.argsort(confidences)
    variants = [variants[i] for i in sorted_indices]
    confidences = [confidences[i] for i in sorted_indices]
    predictions = [predictions[i] for i in sorted_indices]
    
    y_pos = np.arange(len(variants))
    
    # Create horizontal bars with colors based on prediction
    colors = ['C' + str(pred % 10) for pred in predictions]
    bars = plt.barh(y_pos, confidences, color=colors)
    
    # Add the predicted digit as text at the end of each bar
    for i, (bar, pred) in enumerate(zip(bars, predictions)):
        plt.text(bar.get_width() + 1, y_pos[i], f'Digit {pred}', 
                 va='center', fontsize=10)
    
    plt.yticks(y_pos, variants)
    plt.xlabel('Confidence (%)')
    plt.title('Confidence by Preprocessing Strategy')
    plt.xlim(0, 105)  # Leave room for text
    
    # Second plot: Best preprocessing result
    plt.subplot(1, 3, 2)
    plt.imshow(best_result['image_tensor'].cpu().squeeze(), cmap='gray')
    plt.title(f"Best: {strategy_names[best_result['variant']]}\nPrediction: {best_result['predicted_class']} ({best_result['confidence']:.1f}%)")
    plt.axis('off')
    
    # Third plot: TTA result probabilities
    plt.subplot(1, 3, 3)
    bars = plt.bar(range(10), tta_probs.cpu().numpy() * 100)
    bars[tta_digit].set_color('green')
    plt.xlabel('Digit')
    plt.ylabel('Confidence (%)')
    plt.title(f'Test-Time Augmentation\nPrediction: {tta_digit}')
    plt.xticks(range(10))
    plt.ylim(0, 100)
    
    # If we have more than 2 variants, show each preprocessing result
    if len(results) > 2:
        for i, result in enumerate(results):
            plt.subplot(2, len(results), len(results) + i + 1)
            plt.imshow(result['image_tensor'].cpu().squeeze(), cmap='gray')
            plt.title(f"{strategy_names[result['variant']]}\nPred: {result['predicted_class']} ({result['confidence']:.1f}%)")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'prediction_analysis_{os.path.basename(image_path)}.png')
    plt.show()
    
    return best_result['predicted_class'], best_result['confidence']

def ensemble_predict(models, image_tensors, device):
    """
    Make predictions using an ensemble of models for higher accuracy.
    Try each preprocessing variant and pick the one with highest confidence.
    
    Args:
        models: List of trained models
        image_tensors: List of preprocessed image tensors from different strategies
        device: Device to run inference on
        
    Returns:
        Predicted digit, confidence, and best preprocessing strategy index
    """
    # Ensure all models are in evaluation mode
    for model in models:
        model.eval()
    
    best_confidence = -1
    best_prediction = None
    best_variant = None
    best_probs = None
    
    # Try each preprocessing variant
    for variant, image_tensor in enumerate(image_tensors):
        # Move image to device
        image_tensor = image_tensor.to(device)
        
        # Initialize aggregated probabilities for this variant
        aggregated_probs = torch.zeros(10).to(device)
        
        # Get predictions from each model
        with torch.no_grad():
            for model in models:
                _, probabilities = model.predict(image_tensor)
                aggregated_probs += probabilities[0]
        
        # Average the probabilities
        aggregated_probs /= len(models)
        
        # Get the predicted class and confidence
        predicted_class = torch.argmax(aggregated_probs)
        confidence = aggregated_probs[predicted_class].item() * 100
        
        # Check if this is the best so far
        if confidence > best_confidence:
            best_confidence = confidence
            best_prediction = predicted_class.item()
            best_variant = variant
            best_probs = aggregated_probs
    
    return best_prediction, best_confidence, best_variant, best_probs

# --------------------- Test Time Augmentation ---------------------

def test_time_augmentation(model, image_tensor, device, num_augmentations=10):
    """
    Improve prediction accuracy by averaging results over multiple augmented versions 
    of the input image.
    
    Args:
        model: Trained model
        image_tensor: Original image tensor
        device: Device to run inference on
        num_augmentations: Number of augmented versions to create
        
    Returns:
        Predicted digit and aggregated probabilities
    """
    model.eval()
    
    # Define augmentation transforms
    augmentations = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    ])
    
    # Move image to device
    image_tensor = image_tensor.to(device)
    
    # Convert to PIL for augmentation
    image_pil = transforms.ToPILImage()(image_tensor.squeeze(0))
    
    # Initialize aggregated probabilities
    aggregated_probs = torch.zeros(10).to(device)
    
    # Add original image prediction
    with torch.no_grad():
        _, probabilities = model.predict(image_tensor)
        aggregated_probs += probabilities[0]
    
    # Get predictions for augmented versions
    for _ in range(num_augmentations):
        # Apply augmentation
        augmented_pil = augmentations(image_pil)
        
        # Convert back to tensor
        augmented_tensor = transforms.ToTensor()(augmented_pil).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            _, probabilities = model.predict(augmented_tensor)
            aggregated_probs += probabilities[0]
    
    # Average the probabilities
    aggregated_probs /= (num_augmentations + 1)
    
    # Get the predicted class
    predicted_class = torch.argmax(aggregated_probs)
    
    return predicted_class.item(), aggregated_probs

# --------------------- Main Execution ---------------------

if __name__ == "__main__":
    # Load MNIST data
    try:
        print("Loading MNIST dataset...")
        train_images = read_idx_images(TRAIN_IMAGES_PATH)
        train_labels = read_idx_labels(TRAIN_LABELS_PATH)
        test_images = read_idx_images(TEST_IMAGES_PATH)
        test_labels = read_idx_labels(TEST_LABELS_PATH)
        
        # Create datasets with data augmentation for training
        train_dataset = MNISTDataset(train_images, train_labels, augment=True)
        test_dataset = MNISTDataset(test_images, test_labels, augment=False)
        
        # Create data loaders
        batch_size = 128  # Larger batch size for faster training
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4 if torch.cuda.is_available() else 0)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)
        
        # Create model, loss function, and optimizer
        model = ImprovedMNISTNet()
        
        # Use label smoothing for better generalization
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Use AdamW optimizer with weight decay for regularization
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Learning rate scheduler - using OneCycleLR for better convergence
        # This scheduler doesn't require metrics and is called during training
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.003, 
            steps_per_epoch=len(train_dataloader),
            epochs=num_epochs
        )
        
        # Train the model
        num_epochs = 15  # Train for more epochs
        print("\nStarting training...")
        trained_model, history = train_mnist_model(
            train_dataloader, test_dataloader, model, criterion, optimizer, num_epochs, device, scheduler
        )
        
        # Create and train an ensemble of models for better accuracy
        print("\nTraining ensemble models...")
        ensemble_models = []
        
        # Add the best model we just trained
        ensemble_models.append(trained_model)
        
        # Train 2 more models with different random seeds
        for i in range(2):
            print(f"\nTraining ensemble model {i+2}/3...")
            # Set different random seed
            torch.manual_seed(100 + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(100 + i)
            
            # Create a new model with slightly different architecture
            dropout_rate = 0.3 + (i * 0.1)  # Vary dropout rate
            model = ImprovedMNISTNet(dropout_rate=dropout_rate)
            
            # Train for fewer epochs for the ensemble members
            model, _ = train_mnist_model(
                train_dataloader, test_dataloader, model, criterion, optimizer, 
                num_epochs=10, device=device, scheduler=scheduler
            )
            
            ensemble_models.append(model)
            save_model(model, f"ensemble_model_{i+2}.pth")
        
        # Evaluate on the test set with the ensemble
        print("\nEvaluating ensemble on test set...")
        test_iter = iter(test_dataloader)
        images, labels = next(test_iter)
        images, labels = images.to(device), labels.to(device)
        
        ensemble_predictions = []
        # Get ensemble predictions
        for i in range(len(images)):
            pred, _ = ensemble_predict(ensemble_models, images[i:i+1], device)
            ensemble_predictions.append(pred)
        
        ensemble_predictions = torch.tensor(ensemble_predictions, device=device)
        
        # Calculate ensemble accuracy
        ensemble_accuracy = (ensemble_predictions == labels).sum().item() / len(labels) * 100
        print(f"Ensemble Test Accuracy: {ensemble_accuracy:.2f}%")
        
        # Visualize some test results with the ensemble
        visualize_results(images, labels, ensemble_predictions)
        
        # Inference on individual images using the ensemble and test-time augmentation
        print("\nPerforming inference on individual images...")
        image_paths = ["3.png", "5.png", "9.png"]
        
        for image_path in image_paths:
            if os.path.exists(image_path):
                print(f"\nInference for {image_path}:")
                # Preprocess the image with multiple strategies
                image_tensors = preprocess_external_image(image_path)
                
                if image_tensors is not None:
                    # Get prediction with test-time augmentation on single model
                    predicted_digit, confidence = infer_digit(trained_model, image_path, device)
                    
                    # Get ensemble prediction with all preprocessing strategies
                    ensemble_digit, ensemble_confidence, best_variant, ensemble_probs = ensemble_predict(
                        ensemble_models, image_tensors, device
                    )
                    
                    strategy_names = ["Simple Resize", "Centered with Padding", "Adaptive Threshold", "Edge Detection"]
                    print(f"Single model prediction: {predicted_digit} (Confidence: {confidence:.2f}%)")
                    print(f"Ensemble prediction: {ensemble_digit} (Confidence: {ensemble_confidence:.2f}%, Best strategy: {strategy_names[best_variant]})")
                    
                    # Visualize the ensemble prediction
                    plt.figure(figsize=(10, 6))
                    
                    # Show the best image
                    plt.subplot(1, 2, 1)
                    plt.imshow(image_tensors[best_variant].cpu().squeeze(), cmap='gray')
                    plt.title(f'Best Preprocessing: {strategy_names[best_variant]}\nEnsemble Prediction: {ensemble_digit}')
                    plt.axis('off')
                    
                    # Show the confidence for each digit from the ensemble
                    plt.subplot(1, 2, 2)
                    colors = ['lightgray'] * 10
                    colors[ensemble_digit] = 'blue'
                    plt.bar(range(10), ensemble_probs.cpu().numpy() * 100, color=colors)
                    plt.xlabel('Digit')
                    plt.ylabel('Confidence (%)')
                    plt.title(f'Ensemble Confidence: {ensemble_confidence:.1f}%')
                    plt.xticks(range(10))
                    plt.ylim(0, 100)
                    
                    plt.tight_layout()
                    plt.savefig(f'ensemble_prediction_{os.path.basename(image_path)}.png')
                    plt.show()
            else:
                print(f"Image file {image_path} not found.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()