#!/bin/bash

# CUDA Vector Addition Docker setup script

# Create directory
mkdir -p cuda_vector_add_docker
cd cuda_vector_add_docker

# Copy original source code
cp ~/Algorithms/CUDA/Vector_ADD/main.cu .
cp ~/Algorithms/CUDA/Vector_ADD/Makefile .

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source files
COPY main.cu .
COPY Makefile .

# Compile
RUN make

# Run command
CMD ["./main"]
EOF

# Create docker-compose.yml file
cat > docker-compose.yml << 'EOF'
version: '3'

services:
  cuda-vector-add:
    build:
      context: .
      dockerfile: Dockerfile
    image: cuda-vector-add:latest
    container_name: cuda-vector-add
    # Runtime configuration for GPU support
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF

# Create README.md file
cat > README.md << 'EOF'
# CUDA Vector Addition Docker Example

This project is an example of running CUDA vector addition in a Docker container.

## Requirements

- Docker
- NVIDIA Container Toolkit (required for GPU support)

## How to Run

### When using GPU support:

```bash
# Run with Docker Compose
docker-compose up

# Or run Docker command directly
docker build -t cuda-vector-add:latest .
docker run --gpus all cuda-vector-add:latest
```

### Running without GPU (CPU only):

```bash
docker build -t cuda-vector-add:latest .
docker run cuda-vector-add:latest
```

## Notes

- To use GPU support, NVIDIA drivers and NVIDIA Container Toolkit must be installed on the host system.
- CUDA code may not execute in CPU-only mode.

## Docker Image Description

This Docker image includes:
- NVIDIA CUDA 12.0 runtime and development tools
- Vector addition CUDA source code
- Automatic build configuration via Make
EOF

echo "CUDA Vector Addition Docker setup complete."
echo "Build and run the Docker image with the following commands:"
echo ""
echo "  cd cuda_vector_add_docker"
echo "  docker build -t cuda-vector-add:latest ."
echo "  docker run --gpus all cuda-vector-add:latest"
echo ""
echo "For systems without GPU:"
echo "  docker run cuda-vector-add:latest"