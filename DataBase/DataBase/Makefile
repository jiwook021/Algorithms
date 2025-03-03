CXX       := g++
CXXFLAGS  := -std=c++20 -Wall -Wextra -Wpedantic -O2
LDFLAGS   :=

MODULE    := DataBase
SRCS      := $(MODULE).cpp main.cpp
TEST_SRCS := $(MODULE).cpp test_$(MODULE).cpp
OBJS      := $(SRCS:.cpp=.o)
TEST_OBJS := $(TEST_SRCS:.cpp=.o)
TARGET    := main
TEST_BIN  := test_$(MODULE)

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(TEST_BIN) *.o
