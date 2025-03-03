NVCC      := nvcc
NVCCFLAGS := -std=c++20 --gpu-architecture=sm_75 -I/usr/include/x86_64-linux-gnu

MODULE    := CnnMnist
TARGET    := main
TEST_BIN  := test_$(MODULE)

.PHONY: all main test clean

all: $(TARGET)
main: $(TARGET)

$(TARGET): $(MODULE).cu main.cu $(MODULE).cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $(MODULE).cu main.cu -lcudnn -lcublas

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(MODULE).cu test_$(MODULE).cu $(MODULE).cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $(MODULE).cu test_$(MODULE).cu -lcudnn -lcublas

clean:
	rm -f $(TARGET) $(TEST_BIN) *.o
