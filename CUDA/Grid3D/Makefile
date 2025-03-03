NVCC      := nvcc
NVCCFLAGS := -std=c++20 --gpu-architecture=sm_75
CXX       := g++
CXXFLAGS  := -std=c++20 -Wall -Wextra -O2
CUDA_INCLUDE := $(shell [ -d /usr/local/cuda/include ] && echo "-I/usr/local/cuda/include" || echo "")

MODULE    := Grid3D
TARGET    := main
TEST_BIN  := test_$(MODULE)

.PHONY: all main test clean

all: $(TARGET)
main: $(TARGET)

$(TARGET): grid3d.cu grid3d_utils.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ grid3d.cu

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): grid3d_test.cpp grid3d_utils.cuh
	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDE) grid3d_test.cpp -o $@ -lgtest -lgtest_main -pthread

clean:
	rm -f $(TARGET) $(TEST_BIN) *.o
