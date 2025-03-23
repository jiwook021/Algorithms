CXX = g++
CXXFLAGS = -g -Wall -std=c++20 -pthread

all: main

main: main.cpp 
	$(CXX) $(CXXFLAGS) main.cpp -o main -lgtest -lgtest_main -pthread -I/usr/include/opencv4 -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_features2d

clean:
	rm -f main
