# Makefile for YOLOv8 Object Detection and Tracking with CUDA

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -pthread

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin
MODELS_DIR = models

# Files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
EXECUTABLE = $(BIN_DIR)/yolo_tracker

# OpenCV with CUDA
OPENCV_CFLAGS = -I/usr/include
# Don't use pkg-config for libs, specify them manually to avoid missing libraries
OPENCV_LIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_features2d -lopencv_dnn -lopencv_video

# CUDA
CUDA_PATH = /usr/local/cuda
CUDA_CFLAGS = -I$(CUDA_PATH)/include
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas

# Combine flags
CXXFLAGS += $(OPENCV_CFLAGS) $(CUDA_CFLAGS) -I$(INCLUDE_DIR)
LDFLAGS = $(OPENCV_LIBS) $(CUDA_LIBS)

# Targets
.PHONY: all clean directories

all: directories $(EXECUTABLE)

# Create directories
directories:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR) $(MODELS_DIR)

# Main executable
$(EXECUTABLE): $(OBJECTS)
	@echo "Linking $@..."
	@$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Build complete!"

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	@echo "Cleaning build files..."
	@rm -rf $(BUILD_DIR)/* $(BIN_DIR)/*

# Run the application
run: all
	@echo "Running application..."
	@$(EXECUTABLE)

# Run with a specific video file
run-video: all
	@echo "Running with video file..."
	@$(EXECUTABLE) --video $(VIDEO)

# Run with test scenarios
run-tests: all
	@echo "Running test scenarios..."
	@$(EXECUTABLE) --run-tests

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@pkg-config --modversion opencv4 || echo "OpenCV 4 not found!"
	@if [ ! -d $(CUDA_PATH) ]; then echo "CUDA not found at $(CUDA_PATH)!"; fi
	@if [ ! -f $(MODELS_DIR)/yolov8s.onnx ]; then echo "YOLOv8 model not found! Please download it to $(MODELS_DIR)/"; fi

# Download YOLOv8 model
download-model:
	@echo "Downloading YOLOv8s ONNX model..."
	@mkdir -p $(MODELS_DIR)
	@wget -O $(MODELS_DIR)/yolov8s.onnx https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.onnx || \
		(echo "Failed to download model. Please download manually from https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.onnx and place in $(MODELS_DIR)/")

# Show help
help:
	@echo "Available targets:"
	@echo "  all           - Build the application (default)"
	@echo "  clean         - Remove build files"
	@echo "  run           - Build and run the application using the default camera"
	@echo "  run-video     - Build and run with a specific video file (use VIDEO=path/to/video.mp4)"
	@echo "  run-tests     - Run test scenarios for occlusion and fast movement"
	@echo "  check-deps    - Check if all dependencies are installed"
	@echo "  download-model - Download the YOLOv8s ONNX model"
	@echo "  help          - Show this help message"