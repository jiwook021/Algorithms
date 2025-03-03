CXX       := g++
CXXFLAGS  := -std=c++20 -Wall -Wextra -Wpedantic -O2
LDFLAGS   :=
LDFLAGS_TEST := -lgtest -lgtest_main -pthread

MODULE    := PrincipalComponentAnalysis
SRCS      := $(MODULE).cpp main.cpp
TEST_SRCS := $(MODULE).cpp test_$(MODULE).cpp
OBJS      := $(SRCS:.cpp=.o)
TEST_OBJS := $(TEST_SRCS:.cpp=.o)
TARGET    := main
TEST_BIN  := test_$(MODULE)

.PHONY: all run test clean

all: $(TARGET)
run: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDFLAGS_TEST)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(TEST_BIN) *.o
