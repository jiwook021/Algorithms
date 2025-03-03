CXX = g++
CXXFLAGS = -g -Wall -std=c++20 -pthread

all: main

main: main.cpp 
	$(CXX) $(CXXFLAGS) main.cpp -o main `pkg-config --cflags --libs opencv4`

clean:
	rm -f main
