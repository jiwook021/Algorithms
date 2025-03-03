#!/bin/bash

# OpenCV FIR/IIR filter visualization Docker setup script

# Create directory and navigate
mkdir -p opencv_fir_iir_docker
cd opencv_fir_iir_docker

# Copy original source file
cp ~/Algorithms/ComputerVision/openCV/FIR_IIR/main.cpp .
cp ~/Algorithms/ComputerVision/openCV/FIR_IIR/Makefile .

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM ubuntu:22.04

# Install required packages and configure
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    g++ \
    make \
    libopencv-dev \
    python3-opencv \
    libgtk2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source files
COPY main.cpp .
COPY Makefile .

# Libraries for X11 GUI environment setup
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Compile source code
RUN make

# Command to execute when container starts
CMD ["./main"]
EOF

# Create docker-compose.yml file
cat > docker-compose.yml << 'EOF'
version: '3'

services:
  opencv-app:
    build:
      context: .
      dockerfile: Dockerfile
    image: opencv-fir-iir:latest
    container_name: opencv-fir-iir
    # X11 settings for GUI application
    environment:
      - DISPLAY=${DISPLAY}
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
      - .:/app
    # Use host network (easy X11 access)
    network_mode: "host"
EOF

# Create run script
cat > run.sh << 'EOF'
#!/bin/bash

# Allow all access to X11 server
xhost +local:docker

# Run application with Docker Compose
docker-compose up --build

# Restore X11 settings after execution
xhost -local:docker
EOF

# Grant execute permission to run script
chmod +x run.sh

# Create README.md file
cat > README.md << 'EOF'
# OpenCV FIR/IIR Filter Visualization Docker

This project runs an application that visualizes FIR (Finite Impulse Response) and IIR (Infinite Impulse Response) filters using OpenCV in a Docker container.

## Requirements

- Docker
- Docker Compose
- X11 server (for GUI display)

## How to Run

1. Allow X11 access:
   ```bash
   xhost +local:docker
   ```

2. Build and run Docker image:
   ```bash
   docker-compose up --build
   ```
   
   Or use the provided script:
   ```bash
   ./run.sh
   ```

3. Restrict X11 access after execution (optional):
   ```bash
   xhost -local:docker
   ```

## Troubleshooting

### GUI Display Error

If you encounter the following error:
```
Cannot connect to X server
```

Run the following command on the host system to allow Docker X11 access:
```bash
xhost +local:docker
```

### Running on Windows

When using WSL2, additional X11 server configuration may be needed:
1. Install X11 server for Windows (e.g., VcXsrv, Xming)
2. Set environment variable: `export DISPLAY=:0`
EOF

echo "OpenCV FIR/IIR Docker setup complete."
echo "Run with the following command:"
echo ""
echo "  cd opencv_fir_iir_docker"
echo "  ./run.sh"
echo ""
echo "Or"
echo ""
echo "  cd opencv_fir_iir_docker"
echo "  xhost +local:docker"
echo "  docker-compose up --build"