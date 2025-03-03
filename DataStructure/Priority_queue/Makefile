CXX = g++
CXXFLAGS = -g -Wall -Wextra -std=c++23
GTEST_LIBS = -lgtest -lgtest_main -pthread

.PHONY: all test clean

all: main priority_queue_test

main: main.cpp PriorityQueue.cpp PriorityQueue.hpp
	$(CXX) $(CXXFLAGS) main.cpp PriorityQueue.cpp -o main

priority_queue_test: test_PriorityQueue.cpp PriorityQueue.cpp PriorityQueue.hpp
	$(CXX) $(CXXFLAGS) test_PriorityQueue.cpp PriorityQueue.cpp -o priority_queue_test $(GTEST_LIBS)

test: priority_queue_test
	./priority_queue_test

clean:
	$(RM) main priority_queue_test
