import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tqdm import tqdm

"""
Complete Image Classification Exercise with Transfer Learning

This script implements a full image classification pipeline using transfer learning:
1. Data loading and preprocessing
2. Model creation with pretrained networks
3. Training and validation
4. Evaluation on test data
5. Visualization of results

Time Complexity Analysis:
- Training: O(epochs * n * p) where n is number of samples and p is number of parameters
- Inference: O(p) where p is the number of parameters

Space Complexity Analysis:
- O(b * f) where b is batch size and f is feature size
"""

def prepare_data(batch_size=64, num_workers=2):
    """
    Prepare CIFAR-10 dataset with appropriate transformations.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        dict: Dictionary containing data loaders and class names
    """
    print("Preparing CIFAR-10 dataset...")
    
    # Define transformations for training data (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(256),  # Resize to larger dimension before cropping
        transforms.RandomResizedCrop(224),  # ResNet models expect 224x224 input
        transforms.RandomHorizontalFlip(),  # Flip images horizontally with 0.5 probability
        transforms.RandomRotation(10),     # Rotate images randomly +/- 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),  # Color augmentation
        transforms.ToTensor(),  # Convert to tensor (0-1 range)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Define transformations for validation/test data (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # Deterministic crop for evaluation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 dataset
    try:
        # Training dataset
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        
        # Test dataset
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
        
        # Split training set into training and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        # Use fixed random seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        # Get class names
        class_names = train_dataset.classes
        
        print(f"Dataset prepared successfully:")
        print(f"  Training samples: {len(train_subset)}")
        print(f"  Validation samples: {len(val_subset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Number of classes: {len(class_names)}")
        print(f"  Class names: {class_names}")
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'class_names': class_names
        }
        
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return None


def create_model(model_name='resnet18', num_classes=10, feature_extract=True):
    """
    Create a model for transfer learning.
    
    Args:
        model_name (str): Name of the pretrained model to use
        num_classes (int): Number of output classes
        feature_extract (bool): If True, only update the reshaped layer params
        
    Returns:
        model (nn.Module): The neural network model
    """
    print(f"Creating model based on {model_name}...")
    
    # Initialize the model
    model = None
    input_size = 0
    
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        input_size = 224
        
        # Freeze all layers if feature_extract is True
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
                
        # Replace the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(num_ftrs, num_classes)
        )
        
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        input_size = 224
        
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
                
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        input_size = 224
        
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
                
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        input_size = 224
        
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
                
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    else:
        print(f"Invalid model name: {model_name}")
        return None
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Input size: {input_size}x{input_size}")
    
    return model


def train_model(model, data_loaders, criterion=None, optimizer=None, scheduler=None, 
                num_epochs=10, save_dir='models'):
    """
    Train the model with validation.
    
    Args:
        model (nn.Module): The neural network model
        data_loaders (dict): Dictionary of data loaders
        criterion: Loss function (default: CrossEntropyLoss)
        optimizer: Optimizer (default: Adam)
        scheduler: Learning rate scheduler
        num_epochs (int): Number of training epochs
        save_dir (str): Directory to save the best model
        
    Returns:
        model (nn.Module): Trained model
        history (dict): Training history
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set default criterion and optimizer if not provided
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize variables
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Get data loaders
    train_loader = data_loaders['train_loader']
    val_loader = data_loaders['val_loader']
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Progress bar
            loop = tqdm(dataloader, desc=f"{phase.capitalize()}")
            
            # Iterate over data
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                loop.set_postfix(loss=loss.item())
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset) * 100
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
            
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Update learning rate based on validation loss
                if scheduler is not None:
                    scheduler.step(epoch_loss)
                
                # Save best model
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Saved new best model with accuracy: {epoch_acc:.2f}%")
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.8f}")
        print("-" * 50)
    
    # Training complete
    time_elapsed = time.time() - start_time
    print(f"Training completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model weights
    model.load_state_dict(torch.load(best_model_path))
    
    return model, history


def evaluate_model(model, test_loader, class_names):
    """
    Evaluate the model on test data.
    
    Args:
        model (nn.Module): The neural network model
        test_loader (DataLoader): DataLoader for test data
        class_names (list): List of class names
        
    Returns:
        float: Test accuracy
    """
    print("Evaluating model on test data...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Initialize variables
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = list(0. for _ in range(len(class_names)))
    class_total = list(0. for _ in range(len(class_names)))
    
    # Evaluate
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Track loss
            test_loss += loss.item() * inputs.size(0)
            
            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Track class-wise accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            
            # Store predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate overall accuracy
    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Print class-wise accuracy
    print("\nClass-wise accuracy:")
    for i in range(len(class_names)):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"  {class_names[i]}: {class_acc:.2f}%")
    
    # Create confusion matrix
    create_confusion_matrix(all_labels, all_preds, class_names)
    
    return test_accuracy


def create_confusion_matrix(y_true, y_pred, class_names):
    """
    Create and display a confusion matrix.
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        class_names (list): List of class names
    """
    # Create confusion matrix
    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    
    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot raw confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add numbers
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
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add percentages
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, format(cm_norm[i, j], '.2f'),
                    horizontalalignment="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_training_history(history):
    """
    Plot training and validation loss/accuracy.
    
    Args:
        history (dict): Training history
    """
    # Determine number of epochs
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Set up figure
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, test_loader, class_names, num_images=10):
    """
    Visualize model predictions on test images.
    
    Args:
        model (nn.Module): The neural network model
        test_loader (DataLoader): DataLoader for test data
        class_names (list): List of class names
        num_images (int): Number of images to visualize
    """
    # Set device and model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Limit to the specified number of images
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Convert to CPU for numpy operations
    predicted = predicted.cpu()
    probabilities = probabilities.cpu()
    
    # Plot images with predictions
    fig = plt.figure(figsize=(15, 6))
    for idx in range(num_images):
        # Add subplot
        ax = fig.add_subplot(2, (num_images+1)//2, idx+1, xticks=[], yticks=[])
        
        # Convert tensor to numpy array
        img = images[idx].permute(1, 2, 0).numpy()
        
        # Denormalize image for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Display image
        ax.imshow(img)
        
        # Get prediction details
        pred_idx = predicted[idx].item()
        true_idx = labels[idx].item()
        prob = probabilities[idx, pred_idx].item() * 100
        
        # Set title color based on correctness
        title_color = 'green' if pred_idx == true_idx else 'red'
        
        # Set title with prediction and ground truth
        ax.set_title(
            f"Pred: {class_names[pred_idx]} ({prob:.1f}%)\nTrue: {class_names[true_idx]}", 
            color=title_color
        )
    
    plt.tight_layout()
    plt.show()


def run_image_classifier_exercise(model_name='resnet18', num_epochs=5, feature_extract=True):
    """
    Run the complete image classifier exercise.
    
    Args:
        model_name (str): Name of the pretrained model to use
        num_epochs (int): Number of training epochs
        feature_extract (bool): If True, only update the reshaped layer params
        
    Returns:
        str: Completion message
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("===== PRACTICAL EXERCISE: IMAGE CLASSIFIER =====")
    
    # 1. Prepare data
    data = prepare_data(batch_size=64, num_workers=2)
    if data is None:
        return "Exercise failed: Could not prepare data."
    
    # 2. Create model
    model = create_model(model_name=model_name, num_classes=len(data['class_names']), 
                         feature_extract=feature_extract)
    if model is None:
        return "Exercise failed: Could not create model."
    
    # 3. Train model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)
    
    model, history = train_model(
        model=model,
        data_loaders=data,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        save_dir='models'
    )
    
    # 4. Plot training history
    plot_training_history(history)
    
    # 5. Evaluate model on test data
    test_accuracy = evaluate_model(model, data['test_loader'], data['class_names'])
    
    # 6. Visualize predictions
    visualize_predictions(model, data['test_loader'], data['class_names'], num_images=10)
    
    # 7. Save final model
    final_model_path = f'models/cifar10_{model_name}_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return f"Exercise completed successfully! Final test accuracy: {test_accuracy:.2f}%"


if __name__ == "__main__":
    # Run the exercise with ResNet18 and 5 epochs
    result = run_image_classifier_exercise(model_name='resnet18', num_epochs=5, feature_extract=True)
    print(result)
    
    # Optional: Try different models
    # Uncomment to run with different configurations
    # result = run_image_classifier_exercise(model_name='mobilenet_v2', num_epochs=5, feature_extract=True)
    # print(result)
    
    # result = run_image_classifier_exercise(model_name='resnet18', num_epochs=10, feature_extract=False)
    # print(result)