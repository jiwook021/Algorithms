CC = g++
CFLAGS = -g -Wall -std=c++20 -pthread
OPENCV = -I/usr/local/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

all: main

main: main.cpp
	$(CC) $(CFLAGS) main.cpp -o main $(OPENCV)

# Build with address sanitizer to catch memory errors
debug: main.cpp
	$(CC) $(CFLAGS) -fsanitize=address -fno-omit-frame-pointer main.cpp -o main_debug $(OPENCV)

# Build a version that uses a dummy image with no OpenCV dependency
simple: simplified_version.cpp
	$(CC) $(CFLAGS) simplified_version.cpp -o simple_main

clean:
	rm -f main main_debug simple_main

.PHONY: all debug simple clean