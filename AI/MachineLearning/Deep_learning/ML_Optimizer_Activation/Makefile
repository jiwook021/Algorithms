CXX       := g++
CXXFLAGS  := -std=c++20 -Wall -Wextra -Wpedantic -O2
LDFLAGS   := -pthread

TARGET    := main

.PHONY: all main clean

all: $(TARGET)
main: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o
