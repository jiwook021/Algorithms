from ultralytics import YOLO
import cv2
import numpy as np
import time

def detect_objects(image_path, confidence_threshold=0.25):
    """
    Detect objects in an image using YOLOv8.
    
    Args:
        image_path (str): Path to the input image
        confidence_threshold (float): Minimum confidence threshold for detections
        
    Returns:
        tuple: (processed_image, detection_results)
    """
    try:
        # Load the YOLOv8 model
        model = YOLO("yolov8n.pt")  # Load the smallest YOLOv8 model
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image from {image_path}")
        
        # Record the start time for performance measurement
        start_time = time.time()
        
        # Perform detection
        results = model(img, conf=confidence_threshold)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"Processing time: {processing_time:.2f} seconds")
        
        # Get the processed image with bounding boxes
        processed_img = results[0].plot()
        
        return processed_img, results[0]
    
    except Exception as e:
        print(f"Error in object detection: {str(e)}")
        return None, None

def display_results(image, results):
    """
    Display detection results in a more readable format.
    
    Args:
        image: Original or processed image
        results: Detection results from YOLO
    """
    if image is None or results is None:
        print("No results to display")
        return
    
    # Convert BGR to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image with detections
    cv2.imshow("YOLO Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print detailed results
    if hasattr(results, 'boxes'):
        print("\nDetection Results:")
        boxes = results.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            print(f"Object {i+1}:")
            print(f"  Class: {class_name}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Bounding Box: [{x1}, {y1}, {x2}, {y2}]")

# Example usage
if __name__ == "__main__":
    image_path = "sample/image1.png"  # Replace with your image path
    processed_img, results = detect_objects(image_path)
    display_results(processed_img, results)