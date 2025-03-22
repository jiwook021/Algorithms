import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os

class MNISTClassifier(nn.Module):
    """
    CNN for MNIST handwritten digit classification.
    
    Architecture:
    - Two convolutional layers with batch normalization and max pooling
    - Two fully connected layers with dropout
    
    Time Complexity: O(batch_size * channels * height * width) per forward pass
    Memory Complexity: O(batch_size * model_parameters) for storage and computation
    """
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        
        # First conv block: 1 input channel (grayscale), 32 output channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second conv block: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # MNIST images are 28x28
        # After two 2x2 pooling layers: 28 -> 14 -> 7
        # With 64 channels in the last conv layer: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc_bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 10]
        """
        # First conv block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc_bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        device: Device to run on (cuda/cpu)
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        epoch: Current epoch number
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()  # Set model to training mode
    running_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx+1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, device, test_loader, criterion):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model: The neural network model
        device: Device to run on (cuda/cpu)
        test_loader: DataLoader for test data
        criterion: Loss function
        
    Returns:
        tuple: (test_loss, test_accuracy, all_predictions, all_targets)
    """
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():  # No need to track gradients during evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Sum up batch loss
            test_loss += criterion(output, target).item() * data.size(0)
            
            # Get the predictions
            _, pred = torch.max(output, 1)
            
            # Update statistics
            correct += pred.eq(target).sum().item()
            
            # Store predictions and targets for confusion matrix
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate average loss and accuracy
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, test_accuracy, all_predictions, all_targets


def confusion_matrix(y_true, y_pred, num_classes=10):
    """
    Generate confusion matrix from true and predicted labels.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        numpy.ndarray: Confusion matrix
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Set ticks and labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def visualize_results(model, device, test_loader, num_images=10):
    """
    Visualize some test images along with model predictions.
    
    Args:
        model: The neural network model
        device: Device to run on (cuda/cpu)
        test_loader: DataLoader for test data
        num_images: Number of images to visualize
    """
    model.eval()
    
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    with torch.no_grad():
        images_device = images[:num_images].to(device)
        outputs = model(images_device)
        _, predictions = torch.max(outputs, 1)
    
    # Plot the images
    fig = plt.figure(figsize=(15, 4))
    for idx in range(num_images):
        ax = fig.add_subplot(1, num_images, idx+1, xticks=[], yticks=[])
        # Convert image from tensor to numpy and reshape
        img = images[idx].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        
        # Set title with prediction and true label
        pred = predictions[idx].item()
        true_label = labels[idx].item()
        ax.set_title(f"Pred: {pred}\nTrue: {true_label}", 
                     color=("green" if pred == true_label else "red"))
    
    plt.tight_layout()
    plt.show()


def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """
    Plot training and validation history.
    
    Args:
        train_losses: List of training losses per epoch
        train_accs: List of training accuracies per epoch
        val_losses: List of validation losses per epoch
        val_accs: List of validation accuracies per epoch
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_accs, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def mnist_training_tutorial():
    """
    Complete tutorial for training a CNN on the MNIST dataset.
    """
    print("===== TRAINING A CNN ON MNIST DATASET =====")
    
    # 1. Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 2. Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Define transformations for the images
    transform = transforms.Compose([
        transforms.ToTensor(),                     # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,)) # Normalize with MNIST mean and std
    ])
    
    # 4. Load the MNIST dataset
    print("\nLoading MNIST dataset...")
    
    try:
        # Download training data
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        
        # Download test data
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
        
        print(f"MNIST dataset loaded successfully.")
        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")
    except Exception as e:
        print(f"Failed to load MNIST dataset: {e}")
        print("Please check your internet connection or try again later.")
        return "Dataset loading failed."
    
    # 5. Visualize some training examples
    print("\nVisualizing some training examples...")
    examples = iter(train_loader)
    example_data, example_targets = next(examples)
    
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(example_data[i][0], cmap='gray')
        plt.title(f'Label: {example_targets[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # 6. Create the model
    print("\nCreating and initializing the model...")
    model = MNISTClassifier().to(device)
    print(model)
    
    # 7. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    print(f"\nLoss function: {criterion}")
    print(f"Optimizer: {optimizer}")
    print(f"Learning rate scheduler: {scheduler.__class__.__name__}")
    
    # 8. Train the model
    print("\n===== STARTING TRAINING =====")
    num_epochs = 5
    
    # Lists to store metrics for plotting
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluation phase
        val_loss, val_acc, _, _ = evaluate(model, device, test_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f'\nEpoch {epoch} completed in {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Current learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds!")
    
    # 9. Plot training history
    print("\nPlotting training history...")
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # 10. Final evaluation and confusion matrix
    print("\nPerforming final evaluation...")
    test_loss, test_acc, all_preds, all_targets = evaluate(model, device, test_loader, criterion)
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    
    print("\nGenerating confusion matrix...")
    class_names = [str(i) for i in range(10)]  # Digits 0-9
    plot_confusion_matrix(all_targets, all_preds, class_names)
    
    # 11. Visualize some predictions
    print("\nVisualizing model predictions...")
    visualize_results(model, device, test_loader)
    
    # 12. Save the model
    print("\nSaving the trained model...")
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'mnist_cnn.pth'))
    print(f"Model saved to {os.path.join(save_dir, 'mnist_cnn.pth')}")
    
    # 13. Alternative approaches and performance comparison
    print("\n===== ALTERNATIVE APPROACHES =====")
    print("1. Deeper networks (more convolutional layers)")
    print("   - Pros: Can learn more complex features")
    print("   - Cons: More parameters, longer training time, potential overfitting")
    
    print("\n2. Data augmentation techniques")
    print("   - Pros: Increases effective dataset size, improves generalization")
    print("   - Cons: Increased training time, may not be as beneficial for simple datasets like MNIST")
    
    print("\n3. Different optimizers")
    print("   - SGD with momentum: Often works well but requires more tuning")
    print("   - RMSprop: Good for RNNs and non-stationary problems")
    print("   - Adam (used here): Generally performs well in many scenarios")
    
    print("\n4. Ensemble methods")
    print("   - Train multiple models and average their predictions")
    print("   - Pros: Often improves accuracy and robustness")
    print("   - Cons: Increased computational cost and complexity")
    
    return "MNIST training tutorial completed!"


# Run the tutorial if this file is executed directly
if __name__ == "__main__":
    mnist_training_tutorial()