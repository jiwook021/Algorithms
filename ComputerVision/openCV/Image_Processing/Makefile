CXX = g++
TARGET = main
SRC = main.cpp
OPENCV_FLAGS = -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) -o $(TARGET) $(SRC) $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET)