cmake_minimum_required(VERSION 3.10)
project(DeepLearningCV)

# Set C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Path to LibTorch
set(TORCH_PATH "/home/jiwokim/torch/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})

# Force verbose output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

# Match the ABI setting to LibTorch for our main code
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# Display found packages for debugging
message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})  # For our wrapper header
include_directories(${TORCH_INCLUDE_DIRS})

# Build our wrapper library - explicitly include OpenCV here
add_library(opencv_wrapper SHARED opencv_wrapper.cpp)
target_include_directories(opencv_wrapper PRIVATE ${OpenCV_INCLUDE_DIRS})

# This is critical: link against OpenCV directly for the wrapper library
target_link_libraries(opencv_wrapper PRIVATE ${OpenCV_LIBRARIES})

# Image classification example
add_executable(image_classifier deeplearning-cv-cpp.cpp)

# Include directories for the main executable - explicitly exclude OpenCV
target_include_directories(image_classifier PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(image_classifier PRIVATE ${TORCH_INCLUDE_DIRS})

# Link with our wrapper first, then LibTorch (avoid direct OpenCV linking)
target_link_libraries(image_classifier 
    opencv_wrapper
    ${TORCH_LIBRARIES}
)

# Custom CNN example
add_executable(object_detector custom-cnn-cpp.cpp)
target_include_directories(object_detector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(object_detector PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(object_detector 
    opencv_wrapper
    ${TORCH_LIBRARIES}
)