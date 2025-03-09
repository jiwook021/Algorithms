#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p models

# Try multiple sources for compatibility
echo "Attempting to download YOLOv5 model (often more compatible with older OpenCV)..."
wget -O models/yolov5s.onnx https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.onnx

if [ $? -ne 0 ]; then
    echo "Failed to download YOLOv5 model. Trying alternative source..."
    wget -O models/yolov5s.onnx https://drive.google.com/u/0/uc?id=1RYkZwU38ZW8t9BVVHXmZUbg0iB1dO8ml&export=download
fi

# Make sure the file exists and has size
if [ -s models/yolov5s.onnx ]; then
    echo "Successfully downloaded YOLOv5 model to models/yolov5s.onnx"
    echo "Please update your command to use this model:"
    echo "./bin/yolo_tracker --model models/yolov5s.onnx"
else
    echo "Failed to download model. Please download manually."
fi