CXX       := g++
CXXFLAGS  := -std=c++20 -Wall -Wextra -Wpedantic -O2
LDFLAGS   :=

SRCS      := main.cpp p1_library.cpp stats.cpp
OBJS      := $(SRCS:.cpp=.o)
TARGET    := main

.PHONY: all main clean

all: $(TARGET)
main: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o
