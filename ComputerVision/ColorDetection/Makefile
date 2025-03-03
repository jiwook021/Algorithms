CXX = g++
CXXFLAGS = -g -Wall -std=c++20 -pthread

all: main

main: color_detection.cpp
	$(CXX) $(CXXFLAGS) color_detection.cpp -o main `pkg-config --cflags --libs opencv4`

clean:
	rm -f main
