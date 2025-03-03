#!/bin/bash

# Entrypoint script for running the MNIST model
if [ "$MODE" = "train" ]; then
    echo "Starting model training..."
    python mnist_recognition.py --train
elif [ "$MODE" = "inference" ]; then
    echo "Running inference on image: $IMAGE_PATH"
    python mnist_inference.py --image $IMAGE_PATH
else
    echo "Invalid MODE. Use 'train' or 'inference'"
    exit 1
fi
