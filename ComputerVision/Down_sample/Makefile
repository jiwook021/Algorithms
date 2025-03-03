CXX = g++
CXXFLAGS = -g -Wall -std=c++20 -pthread

all: main

main: main.cpp stb_image.h stb_image_write.h
	$(CXX) $(CXXFLAGS) main.cpp -o main

clean:
	rm -f main
