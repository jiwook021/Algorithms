CXX       := g++
CXXFLAGS  := -std=c++20 -Wall -Wextra -O2

MODULE    := SinSumCpu
TARGET    := main
TEST_BIN  := test_$(MODULE)

.PHONY: all main test clean

all: $(TARGET)
main: $(TARGET)

$(TARGET): main.cpp $(MODULE).hpp
	$(CXX) $(CXXFLAGS) -o $@ main.cpp

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): test_$(MODULE).cpp $(MODULE).hpp
	$(CXX) $(CXXFLAGS) -o $@ test_$(MODULE).cpp

clean:
	rm -f $(TARGET) $(TEST_BIN) *.o
