CXX       := g++
CXXFLAGS  := -std=c++20 -Wall -Wextra -O2
LDFLAGS   :=

MODULE    := LeastCommonAncestor
SRCS      := $(MODULE).cpp main.cpp
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
