CXX       := g++
CXXFLAGS  := -std=c++20 -Wall -Wextra -O2 -DTESTING
LDFLAGS   := $(shell pkg-config --libs opencv4 2>/dev/null || echo "-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs")
CXXFLAGS  += $(shell pkg-config --cflags opencv4 2>/dev/null)

MODULE    := MedianFilterNoise
SRCS      := $(MODULE).cpp main.cpp
TEST_SRCS := $(MODULE).cpp test_$(MODULE).cpp
OBJS      := $(SRCS:.cpp=.o)
TEST_OBJS := $(TEST_SRCS:.cpp=.o)
TARGET    := main
TEST_BIN  := test_$(MODULE)

.PHONY: all main test clean

all: $(TARGET)
main: $(TARGET)

$(TARGET): $(MODULE).cpp main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgtest -lgtest_main -pthread

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(MODULE).cpp test_$(MODULE).cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgtest -lgtest_main -pthread

clean:
	rm -f $(TARGET) $(TEST_BIN) *.o
