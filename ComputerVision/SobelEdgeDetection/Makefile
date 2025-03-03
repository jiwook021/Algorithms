CXX = g++
CXXFLAGS = -g -Wall -std=c++20 -pthread

all: main

main: sobel_edge.cpp
	$(CXX) $(CXXFLAGS) sobel_edge.cpp -o main `pkg-config --cflags --libs opencv4`

clean:
	rm -f main
