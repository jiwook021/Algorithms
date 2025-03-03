CXX       := g++
CXXFLAGS  := -std=c++20 -Wall -Wextra -Wpedantic -O2
LDFLAGS   := -pthread -lm

MODULE    := RLCCircuit
SRCS      := PIDController.cpp CircuitModel.cpp CircuitSimulator.cpp
MAIN_SRCS := $(SRCS) main.cpp
TEST_SRCS := $(SRCS) test_$(MODULE).cpp
DEPS      := PIDController.hpp CircuitModel.hpp CircuitSimulator.hpp
MAIN_OBJS := $(MAIN_SRCS:.cpp=.o)
TEST_OBJS := $(TEST_SRCS:.cpp=.o)
TARGET    := $(MODULE)
TEST_BIN  := test_$(MODULE)

.PHONY: all main test clean

all: $(TARGET)
main: $(TARGET)

$(TARGET): $(MAIN_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -lgtest -lgtest_main

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(TEST_BIN) *.o *.csv
