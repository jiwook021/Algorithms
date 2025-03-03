CXX       := g++
CXXFLAGS  := -std=c++20 -Wall -Wextra -O2
OPENCV_FLAGS := $(shell pkg-config --cflags --libs opencv4) -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_features2d -lopencv_calib3d

MODULE    := OrbSlam
TARGET    := main
TEST_BIN  := test_$(MODULE)

.PHONY: all main test clean

all: $(TARGET)
main: $(TARGET)

$(TARGET): $(MODULE).cpp main.cpp $(MODULE).hpp
	$(CXX) $(CXXFLAGS) -o $@ $(MODULE).cpp main.cpp $(OPENCV_FLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(MODULE).cpp test_$(MODULE).cpp $(MODULE).hpp
	$(CXX) $(CXXFLAGS) -o $@ $(MODULE).cpp test_$(MODULE).cpp $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET) $(TEST_BIN) *.o
