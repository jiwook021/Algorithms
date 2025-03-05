CXX = g++
TARGET = main
SRC = main.cpp
OPENCV_FLAGS = -I/usr/include/opencv4 -I/usr/include/opencv4  -lopencv_photo -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_features2d
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) -o $(TARGET) $(SRC) $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET)