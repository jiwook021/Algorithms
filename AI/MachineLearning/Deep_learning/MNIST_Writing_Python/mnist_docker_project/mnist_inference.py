import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
from skimage.filters import threshold_otsu, threshold_local
from skimage import feature
from torchvision import transforms

# Define the same model architecture
class ImprovedMNISTNet(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super(ImprovedMNISTNet, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fourth Convolutional Block
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Calculated flattened size
        flattened_size = 256 * 7 * 7
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)
    
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # Second convolutional block
        x_res = x  # Save for residual connection
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        # Add residual connection
        if x_res.size(1) != x.size(1):
            x_res = F.pad(x_res, (0, 0, 0, 0, 0, x.size(1) - x_res.size(1)))
        x = x + x_res  # Residual connection
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        
        # Third convolutional block
        x_res = x  # Save for residual connection
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        # Add residual connection
        if x_res.size(1) != x.size(1):
            x_res = F.pad(x_res, (0, 0, 0, 0, 0, x.size(1) - x_res.size(1)))
        x = x + x_res  # Residual connection
        
        # Fourth convolutional block
        x_res = x  # Save for residual connection
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        # Add residual connection
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
        """Perform prediction with highest class probability"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities

def preprocess_image(image_path):
    """
    Apply multiple preprocessing strategies for the image
    """
    try:
        # Load image
        original_image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Display original image
        plt.figure(figsize=(5, 5))
        plt.imshow(original_image, cmap='gray')
        plt.title('Original image')
        plt.axis('off')
        plt.savefig(f'original_{os.path.basename(image_path)}.png')
        
        # Generate various preprocessing strategies
        preprocessed_images = []
        titles = []
        
        # Strategy 1: Simple resize (28x28)
        # Strategy 1: Simple resize (28x28)
        image_array1 = np.array(image1).astype(np.float32) / 255.0
        # Check if digit is black (0) on white background (1) - MNIST standard
        if np.mean(image_array1) < 0.5:
            image_array1 = 1.0 - image_array1
        preprocessed_images.append(image_array1)
        titles.append("Simple resize")
        
        # Strategy 2: Center digit with bounding box
        image2 = original_image.copy()
        image_array2 = np.array(image2)
        
        try:
            # Binarize with Otsu's thresholding
            thresh = threshold_otsu(image_array2)
            binary_image = image_array2 < thresh  # Assume digit is darker than background
        except:
            # Fall back to simple thresholding if Otsu fails
            binary_image = image_array2 < 128
        
        # Find bounding box of the digit
        rows = np.any(binary_image, axis=1)
        cols = np.any(binary_image, axis=0)
        
        if np.any(rows) and np.any(cols):
            # Check non-zero region (digit)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Maintain aspect ratio
            width = x_max - x_min
            height = y_max - y_min
            
            # Add padding
            padding = max(width, height) // 4
            y_min = max(0, y_min - padding)
            y_max = min(image_array2.shape[0] - 1, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(image_array2.shape[1] - 1, x_max + padding)
            
            # Crop digit with padding
            image2 = image2.crop((x_min, y_min, x_max, y_max))
            
            # Create square image to maintain aspect ratio
            size = max(image2.width, image2.height)
            square_image = Image.new('L', (size, size), 255)
            # Center the digit in the square
            paste_x = (size - image2.width) // 2
            paste_y = (size - image2.height) // 2
            square_image.paste(image2, (paste_x, paste_y))
            
            # Resize to 20x20
            image2 = square_image.resize((20, 20), Image.LANCZOS)
            
            # Create new image with white background
            padded_image = Image.new('L', (28, 28), 255)
            # Center the digit
            padded_image.paste(image2, ((28 - 20) // 2, (28 - 20) // 2))
            image2 = padded_image
        else:
            # If no foreground detected, resize to 28x28
            image2 = image2.resize((28, 28), Image.LANCZOS)
        
        image_array2 = np.array(image2).astype(np.float32) / 255.0
        # Invert if needed (MNIST uses white background, black digits)
        if np.mean(image_array2) < 0.5:
            image_array2 = 1.0 - image_array2
        preprocessed_images.append(image_array2)
        titles.append("Centered with padding")
        
        # Strategy 3: Adaptive thresholding for noise handling
        image3 = original_image.copy()
        image_array3 = np.array(image3)
        
        try:
            # Apply adaptive thresholding
            block_size = max(11, min(image_array3.shape) // 10 * 2 + 1)  # Must be odd
            thresh = threshold_local(image_array3, block_size=block_size, method='gaussian')
            binary_image = image_array3 < thresh
        except:
            # Fall back to global thresholding
            binary_image = image_array3 < np.mean(image_array3)
        
        # Convert to image
        binary_array = (binary_image.astype(np.uint8) * 255)
        binary_image = Image.fromarray(binary_array)
        
        # Resize to 28x28
        image3 = binary_image.resize((28, 28), Image.LANCZOS)
        image_array3 = np.array(image3).astype(np.float32) / 255.0
        
        # Invert if needed
        if np.mean(image_array3) < 0.5:
            image_array3 = 1.0 - image_array3
        preprocessed_images.append(image_array3)
        titles.append("Adaptive thresholding")
        
        # Strategy 4: Edge detection to focus on digit shape
        try:
            # Use Canny edge detection
            image_array4 = np.array(original_image)
            edges = feature.canny(image_array4 / 255.0, sigma=1)
            edge_image = Image.fromarray((edges * 255).astype(np.uint8))
            
            # Resize to 28x28
            image4 = edge_image.resize((28, 28), Image.LANCZOS)
            image_array4 = np.array(image4).astype(np.float32) / 255.0
            
            # Invert if needed
            if np.mean(image_array4) < 0.5:
                image_array4 = 1.0 - image_array4
            preprocessed_images.append(image_array4)
            titles.append("Edge detection")
        except:
            # Skip if edge detection fails
            pass
        
        # Visualize all preprocessing variants
        plt.figure(figsize=(15, 4))
        for i, (img_array, title) in enumerate(zip(preprocessed_images, titles)):
            plt.subplot(1, len(preprocessed_images), i+1)
            plt.imshow(img_array, cmap='gray')
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'preprocessed_{os.path.basename(image_path)}.png')
        
        # Convert all variants to tensors
        image_tensors = []
        for img_array in preprocessed_images:
            # Convert to tensor and add batch and channel dimensions
            tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            image_tensors.append(tensor)
        
        return image_tensors
    
    except Exception as e:
        print(f"Image processing error {image_path}: {e}")
        return None

def ensemble_predict(models, image_tensors, device):
    """
    Prediction using ensemble model
    Try each preprocessing variant and select the one with highest confidence
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
        
        # Compute average probabilities
        aggregated_probs /= len(models)
        
        # Get predicted class and confidence
        predicted_class = torch.argmax(aggregated_probs)
        confidence = aggregated_probs[predicted_class].item() * 100
        
        # Check if this is the best so far
        if confidence > best_confidence:
            best_confidence = confidence
            best_prediction = predicted_class.item()
            best_variant = variant
            best_probs = aggregated_probs
    
    return best_prediction, best_confidence, best_variant, best_probs

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='MNIST digit recognition')
    parser.add_argument('--image', type=str, required=True, help='Path to image to recognize')
    args = parser.parse_args()
    
    print(f"Using device: {device}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Attempt to load models
    models = []
    model_files = [
        'best_mnist_model.pth',
        'mnist_model.pth',
        'ensemble_model_2.pth',
        'ensemble_model_3.pth'
    ]
    
    # Load available model files
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model = ImprovedMNISTNet()
                model.load_state_dict(torch.load(model_file, map_location=device))
                model.to(device)
                model.eval()
                models.append(model)
                print(f"Model loaded: {model_file}")
            except Exception as e:
                print(f"Model loading error {model_file}: {e}")
    
    if not models:
        # Instantiate a new model if no models found
        model = ImprovedMNISTNet()
        model.to(device)
        model.eval()
        models.append(model)
        print("Warning: No saved models found, using an uninitialized model.")
    
    # Image preprocessing
    image_tensors = preprocess_image(args.image)
    
    if image_tensors is not None:
        # Perform ensemble prediction
        if len(models) > 1:
            predicted_digit, confidence, best_variant, probs = ensemble_predict(models, image_tensors, device)
            # Visualize results
            strategy_names = ["Simple resize", "Centered with padding", "Adaptive thresholding", "Edge detection"]
            print(f"Ensemble prediction: {predicted_digit} (Confidence: {confidence:.2f}%, Best strategy: {strategy_names[best_variant] if best_variant < len(strategy_names) else 'Unknown'})")
            
            # Define colors
            plt.figure(figsize=(10, 6))
            colors = ['lightgray'] * 10
            colors[predicted_digit] = 'blue'
            
            # Save result graph
            plt.subplot(1, 2, 1)
            plt.imshow(image_tensors[best_variant].cpu().squeeze(), cmap='gray')
            plt.title(f'Best preprocessing: {strategy_names[best_variant] if best_variant < len(strategy_names) else "Unknown"}\nPrediction: {predicted_digit}')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.bar(range(10), probs.cpu().numpy() * 100, color=colors)
            plt.xlabel('Digit')
            plt.ylabel('Confidence (%)')
            plt.title(f'Ensemble confidence: {confidence:.1f}%')
            plt.xticks(range(10))
            plt.ylim(0, 100)
            
            plt.tight_layout()
            plt.savefig(f'comparison_{os.path.basename(args.image)}.png')
            
        else:
            # Single model prediction
            model = models[0]
            best_confidence = -1
            best_prediction = None
            best_variant = None
            
            for variant, image_tensor in enumerate(image_tensors):
                image_tensor = image_tensor.to(device)
                with torch.no_grad():
                    predicted_class, probabilities = model.predict(image_tensor)
                    confidence = probabilities[0][predicted_class.item()].item() * 100
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_prediction = predicted_class.item()
                        best_variant = variant
            
            strategy_names = ["Simple resize", "Centered with padding", "Adaptive thresholding", "Edge detection"]
            print(f"Single model prediction: {best_prediction} (Confidence: {best_confidence:.2f}%, Best strategy: {strategy_names[best_variant] if best_variant < len(strategy_names) else 'Unknown'})")
            
            # Save result graph
            plt.figure(figsize=(6, 6))
            plt.imshow(image_tensors[best_variant].cpu().squeeze(), cmap='gray')
            plt.title(f'Best preprocessing: {strategy_names[best_variant] if best_variant < len(strategy_names) else "Unknown"}\nPrediction: {best_prediction} (Confidence: {best_confidence:.1f}%)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'prediction_{os.path.basename(args.image)}.png')
    else:
        print(f"Cannot process image: {args.image}")

if __name__ == "__main__":
    main()
