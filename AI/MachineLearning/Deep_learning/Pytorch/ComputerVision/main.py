import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import time
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.utils import make_grid, draw_segmentation_masks, draw_bounding_boxes

class ResNet18Transfer(nn.Module):
    """
    Transfer learning model based on pre-trained ResNet18 for image classification.
    
    Time Complexity: O(batch_size * channels * height * width) per forward pass
    Memory Complexity: O(batch_size * params) for storing activations and gradients
    """
    def __init__(self, num_classes=10, freeze_backbone=True):
        """
        Initialize the model with a pre-trained ResNet18 backbone.
        
        Args:
            num_classes (int): Number of output classes
            freeze_backbone (bool): Whether to freeze the backbone weights
        """
        super(ResNet18Transfer, self).__init__()
        
        # Load pre-trained ResNet18 model
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Freeze the backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        # ResNet18's fc layer has input features = 512
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        return self.backbone(x)

class ObjectDetectionDemo:
    """
    Class to demonstrate object detection using a pre-trained Faster R-CNN model.
    """
    def __init__(self, device):
        """
        Initialize the object detection model.
        
        Args:
            device (torch.device): Device to run the model on
        """
        self.device = device
        
        # Load a pre-trained Faster R-CNN model
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.to(device)
        self.model.eval()
        
        # Get the class labels
        self.categories = weights.meta["categories"]
    
    def predict(self, image):
        """
        Run object detection on a single image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            tuple: (processed_image, predictions)
        """
        # Transform the image
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(image).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model([img_tensor])[0]
        
        return img_tensor, prediction
    
    def visualize_detection(self, image, prediction, score_threshold=0.7):
        """
        Visualize detection results.
        
        Args:
            image (torch.Tensor): Image tensor
            prediction (dict): Prediction dictionary from the model
            score_threshold (float): Confidence threshold for showing predictions
            
        Returns:
            PIL.Image: Image with bounding boxes
        """
        # Convert image tensor to uint8 for drawing
        image_uint8 = (255.0 * image.permute(1, 2, 0)).byte().cpu()
        
        # Filter predictions by score
        keep_idx = prediction['scores'] > score_threshold
        boxes = prediction['boxes'][keep_idx].cpu()
        labels = prediction['labels'][keep_idx].cpu()
        scores = prediction['scores'][keep_idx].cpu()
        
        # Create category labels with scores
        result_labels = [f"{self.categories[label]}: {score:.2f}" 
                        for label, score in zip(labels, scores)]
        
        # Draw bounding boxes
        result_image = draw_bounding_boxes(
            image=image_uint8,
            boxes=boxes,
            labels=result_labels,
            width=2,
            colors="red"
        )
        
        # Convert tensor to PIL Image for display
        result_image = Image.fromarray(result_image.permute(1, 2, 0).numpy())
        return result_image

class SegmentationDemo:
    """
    Class to demonstrate image segmentation using a pre-trained FCN model.
    """
    def __init__(self, device):
        """
        Initialize the segmentation model.
        
        Args:
            device (torch.device): Device to run the model on
        """
        self.device = device
        
        # Load pre-trained FCN model
        weights = FCN_ResNet50_Weights.DEFAULT
        self.model = fcn_resnet50(weights=weights)
        self.model.to(device)
        self.model.eval()
        
        # Get class information
        self.categories = weights.meta["categories"]
    
    def predict(self, image):
        """
        Run segmentation on a single image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            tuple: (processed_image, output_mask)
        """
        # Transform the image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        img_tensor = transform(image).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model([img_tensor])["out"][0]
        
        # Get normalized image (without normalization) for visualization
        img_vis = transforms.ToTensor()(image).to(self.device)
        
        return img_vis, output
    
    def visualize_segmentation(self, image, output, score_threshold=0.7):
        """
        Visualize segmentation results.
        
        Args:
            image (torch.Tensor): Image tensor
            output (torch.Tensor): Model output tensor
            score_threshold (float): Confidence threshold for showing predictions
            
        Returns:
            PIL.Image: Image with segmentation masks
        """
        # Get class predictions and confidence scores
        normalized_masks = torch.nn.functional.softmax(output, dim=0)
        boolean_masks = normalized_masks > score_threshold
        
        # Skip background class (index 0)
        if boolean_masks.shape[0] > 1:
            boolean_masks = boolean_masks[1:]
        else:
            # If only background class is detected, return original image
            return Image.fromarray((255 * image.permute(1, 2, 0)).byte().cpu().numpy())
        
        # Draw segmentation masks
        result_image = draw_segmentation_masks(
            image=(255 * image).byte().cpu(),
            masks=boolean_masks.cpu(),
            alpha=0.5
        )
        
        # Convert tensor to PIL Image for display
        result_image = Image.fromarray(result_image.permute(1, 2, 0).numpy())
        return result_image

def visualize_filters(model, layer_name='conv1', num_filters=8):
    """
    Visualize filters from a convolutional layer.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer to visualize
        num_filters: Number of filters to visualize
    """
    # Extract the specified layer
    layer = None
    if hasattr(model, layer_name):
        layer = getattr(model, layer_name)
    elif layer_name == 'conv1' and hasattr(model, 'backbone'):
        # Handle ResNet structure
        layer = model.backbone.conv1
    
    if layer is None or not isinstance(layer, nn.Conv2d):
        print(f"Layer {layer_name} not found or not a Conv2d layer")
        return
    
    # Get the filters (weights)
    filters = layer.weight.data.cpu()
    
    # Limit the number of filters to visualize
    num_filters = min(num_filters, filters.shape[0])
    
    # Create a grid of filter visualizations
    fig, axes = plt.subplots(1, num_filters, figsize=(20, 4))
    
    for i in range(num_filters):
        # Get the filter at index i
        filter_i = filters[i, 0]  # Show only first channel for multi-channel filters
        
        # Normalize filter values to 0-1 range for visualization
        filter_min, filter_max = filter_i.min(), filter_i.max()
        filter_norm = (filter_i - filter_min) / (filter_max - filter_min + 1e-8)
        
        # Display the filter
        axes[i].imshow(filter_norm, cmap='viridis')
        axes[i].set_title(f'Filter {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_feature_maps(model, image, layer_name='layer1'):
    """
    Visualize feature maps from a specific layer when processing an image.
    
    Args:
        model: PyTorch model
        image: Input image tensor [C,H,W]
        layer_name: Name of the layer to visualize
    """
    # Create a hook function to capture outputs
    feature_maps = []
    
    def hook_function(module, input, output):
        feature_maps.append(output)
    
    # Register the hook
    hook = None
    if hasattr(model, layer_name):
        hook = getattr(model, layer_name).register_forward_hook(hook_function)
    elif hasattr(model, 'backbone') and hasattr(model.backbone, layer_name):
        hook = getattr(model.backbone, layer_name).register_forward_hook(hook_function)
    
    if hook is None:
        print(f"Layer {layer_name} not found")
        return
    
    # Forward pass with the image
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        image_batch = image.unsqueeze(0)
        _ = model(image_batch)
    
    # Remove the hook
    hook.remove()
    
    # Check if we have feature maps
    if not feature_maps:
        print("No feature maps captured")
        return
    
    # Get feature maps from the first (and only) forward pass
    feature_map = feature_maps[0][0]  # First image in batch
    
    # Determine number of feature maps to show
    num_features = min(16, feature_map.shape[0])
    
    # Create a grid to visualize the feature maps
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_features):
        # Get a single feature map
        feature = feature_map[i].cpu()
        
        # Normalize the feature map
        feature_min, feature_max = feature.min(), feature.max()
        feature = (feature - feature_min) / (feature_max - feature_min + 1e-8)
        
        # Display the feature map
        axes[i].imshow(feature, cmap='viridis')
        axes[i].set_title(f'Feature {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def apply_gradcam(model, image, target_class=None):
    """
    Apply Grad-CAM to visualize the regions of an image the model focuses on.
    
    Args:
        model: PyTorch model
        image: Input image tensor [C,H,W]
        target_class: Target class index (if None, uses the predicted class)
        
    Returns:
        tuple: (original_image, heatmap, overlaid_image)
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Add batch dimension
    image_batch = image.unsqueeze(0)
    
    # Get the model output
    image_batch.requires_grad_()
    
    # Find the last convolutional layer
    target_layer = None
    
    # For ResNet, the last conv layer is the last block's conv3
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'):
        target_layer = model.backbone.layer4[-1].conv3
    
    if target_layer is None:
        print("Could not find a suitable target layer for Grad-CAM")
        return None, None, None
    
    # Store activation maps and gradients
    activations = []
    gradients = []
    
    def save_activation(module, input, output):
        activations.append(output.detach())
    
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    # Register hooks
    handle_fwd = target_layer.register_forward_hook(save_activation)
    handle_bwd = target_layer.register_backward_hook(save_gradient)
    
    # Forward pass
    output = model(image_batch)
    
    # Determine the target class
    if target_class is None:
        # Use the predicted class
        target_class = output.argmax(dim=1).item()
    
    # Zero the gradients
    model.zero_grad()
    
    # Compute gradients with respect to the target class
    output[0, target_class].backward()
    
    # Remove hooks
    handle_fwd.remove()
    handle_bwd.remove()
    
    # Process the activations and gradients
    activations = activations[0]
    gradients = gradients[0]
    
    # Global average pooling of gradients
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    
    # Weight the activation maps
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    
    # Apply ReLU to focus on positive contributions
    cam = torch.nn.functional.relu(cam)
    
    # Resize to original image size
    cam = torch.nn.functional.interpolate(
        cam, size=(image.shape[1], image.shape[2]), 
        mode='bilinear', align_corners=False
    )
    
    # Normalize
    cam = cam[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Convert to numpy for visualization
    image_np = image.permute(1, 2, 0).cpu().numpy()
    cam_np = cam.cpu().numpy()
    
    # Create heatmap
    heatmap = plt.cm.jet(cam_np)[:, :, :3]
    
    # Overlay heatmap on image
    overlaid = image_np * 0.7 + heatmap * 0.3
    
    return image_np, heatmap, overlaid

def computer_vision_tutorial():
    """
    Demonstrates computer vision tasks using PyTorch.
    """
    print("===== COMPUTER VISION WITH PYTORCH =====")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Transfer Learning for Image Classification
    print("\n===== TRANSFER LEARNING FOR IMAGE CLASSIFICATION =====")
    
    # Create a model with pre-trained weights
    model = ResNet18Transfer(num_classes=10, freeze_backbone=True)
    model = model.to(device)
    
    print(f"Created transfer learning model based on ResNet18")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Explain transfer learning
    print("\nTransfer Learning Benefits:")
    print("1. Faster convergence: Pre-trained models already know basic features")
    print("2. Better performance: Especially useful with limited data")
    print("3. Resource efficiency: Requires less computation for training")
    
    print("\nCommon Transfer Learning Strategies:")
    print("1. Feature extraction: Freeze backbone, train only new classifier")
    print("2. Fine-tuning: Train entire network with lower learning rate")
    print("3. Progressive unfreezing: Gradually unfreeze layers during training")
    
    # 2. Data Augmentation
    print("\n===== DATA AUGMENTATION =====")
    
    # Create an example image for augmentation demonstration
    dummy_image = torch.ones(3, 224, 224)  # RGB image
    
    # Define augmentation transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    print("Data Augmentation Benefits:")
    print("1. Increases effective dataset size")
    print("2. Improves model generalization")
    print("3. Reduces overfitting")
    print("4. Helps model become invariant to transformations")
    
    print("\nCommon Augmentation Techniques:")
    print("1. Geometric: Flips, rotations, crops, scaling")
    print("2. Color: Brightness, contrast, saturation adjustments")
    print("3. Noise: Adding Gaussian or salt-and-pepper noise")
    print("4. Cutout/Mixup: Advanced techniques that mask or blend images")
    
    # 3. Object Detection
    print("\n===== OBJECT DETECTION =====")
    
    print("Object Detection Categories:")
    print("1. Single-stage detectors (e.g., YOLO, SSD):")
    print("   - Faster but generally less accurate")
    print("   - Good for real-time applications")
    
    print("\n2. Two-stage detectors (e.g., Faster R-CNN):")
    print("   - Region proposal + classification")
    print("   - More accurate but slower")
    
    print("\nPyTorch provides pre-trained models through torchvision:")
    print("- Faster R-CNN")
    print("- SSD")
    print("- RetinaNet")
    print("- YOLO variants (via community implementations)")
    
    # 4. Semantic Segmentation
    print("\n===== SEMANTIC SEGMENTATION =====")
    
    print("Semantic Segmentation assigns a class to each pixel in an image")
    
    print("\nPopular architectures implemented in torchvision:")
    print("1. FCN (Fully Convolutional Networks)")
    print("2. DeepLabV3")
    print("3. LR-ASPP (Lite Reduced Atrous Spatial Pyramid Pooling)")
    
    print("\nCommon evaluation metrics:")
    print("- IoU (Intersection over Union)")
    print("- Pixel Accuracy")
    print("- Mean IoU across all classes")
    
    # 5. Model Interpretation
    print("\n===== MODEL INTERPRETATION =====")
    
    print("Techniques for understanding what CNN models focus on:")
    print("1. Filter visualization: Examine what patterns each filter detects")
    print("2. Feature map visualization: See activations for a given input")
    print("3. Grad-CAM: Highlight regions important for classification")
    print("4. Occlusion sensitivity: Observe how masking regions affects predictions")
    
    # 6. Tips and Best Practices
    print("\n===== COMPUTER VISION BEST PRACTICES =====")
    
    print("1. Start with pre-trained models whenever possible")
    print("2. Use appropriate data augmentation for your task")
    print("3. Ensure balanced class distribution in the dataset")
    print("4. Monitor both training and validation metrics")
    print("5. Use appropriate evaluation metrics for your task")
    print("6. Consider ensemble methods for critical applications")
    print("7. Analyze failure cases to improve the model")
    print("8. Test with real-world data in the target environment")
    
    return "Computer vision tutorial completed!"

# Run the tutorial if this file is executed directly
if __name__ == "__main__":
    computer_vision_tutorial()