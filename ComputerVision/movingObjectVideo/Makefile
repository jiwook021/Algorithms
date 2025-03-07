CXX = g++
CXXFLAGS = -g -Wall -std=c++20 -pthread

# OpenCV include directory
OPENCV_INCLUDE = -I/usr/include/opencv4

# OpenCV libraries needed for this project
# Added opencv_calib3d which contains findHomography
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
              -lopencv_videoio -lopencv_features2d -lopencv_calib3d

all: main

main: main.cpp
	$(CXX) $(CXXFLAGS) $(OPENCV_INCLUDE) $< -o $@ $(OPENCV_LIBS)

clean:
	rm -f main