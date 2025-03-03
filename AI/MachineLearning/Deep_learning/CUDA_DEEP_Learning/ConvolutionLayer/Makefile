NVCC      := nvcc
NVCCFLAGS := -std=c++20 --gpu-architecture=sm_75
OPENCV_FLAGS := -ccbin g++ -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

MODULE    := ConvolutionLayer
TARGET    := main
TEST_BIN  := test_$(MODULE)

.PHONY: all main test clean

all: $(TARGET)
main: $(TARGET)

$(TARGET): $(MODULE).cu main.cu $(MODULE).cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $(MODULE).cu main.cu $(OPENCV_FLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(MODULE).cu test_$(MODULE).cu $(MODULE).cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $(MODULE).cu test_$(MODULE).cu

clean:
	rm -f $(TARGET) $(TEST_BIN) *.o
