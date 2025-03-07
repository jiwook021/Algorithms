CXX = g++
CXXFLAGS = -g -Wall -std=c++20 -pthread

# Get only the required OpenCV libraries instead of all of them
OPENCV_LIBS = $(shell pkg-config --libs-only-l opencv4 | sed 's/-lopencv_gapi//')

# Get the include and library paths
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LDFLAGS = $(shell pkg-config --libs-only-L --libs-only-other opencv4)

all: main

main: main.cpp
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) main.cpp -o main $(OPENCV_LDFLAGS) $(OPENCV_LIBS)

clean:
	rm -f main