CC = g++
CFLAGS = -g -Wall -std=c++17 \
         -I/home/jiwokim/torch/libtorch/include \
         -I/home/jiwokim/torch/libtorch/include/torch/csrc/api/include \
         -D_GLIBCXX_USE_CXX11_ABI=0 \
         -pthread -fopenmp -O3 -mavx2 -mfma

LDFLAGS = -L/home/jiwokim/torch/libtorch/lib \
          -ltorch -ltorch_cpu -lc10 \
          -Wl,-rpath,/home/jiwokim/torch/libtorch/lib

TARGET = main
SRC = main.cpp

all: $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -L/home/jiwokim/torch/libtorch/lib $(LIBS) $(INCLUDES) -ltorch -ltorch_cpu -lc10 -pthread -o main

# Optimized build target - uses aggressive optimization flags
optimized: $(SRC)
	$(CC) $(CFLAGS) -O3 -march=native -ffast-math $(SRC) $(LDFLAGS) -L/home/jiwokim/torch/libtorch/lib $(LIBS) $(INCLUDES) -ltorch -ltorch_cpu -lc10 -pthread -o main

# AVX2 build target - specifically optimized for AVX2-compatible CPUs
avx2: $(SRC)
	$(CC) $(CFLAGS) -O3 -mavx2 -mfma -march=native -ffast-math $(SRC) $(LDFLAGS) -L/home/jiwokim/torch/libtorch/lib $(LIBS) $(INCLUDES) -ltorch -ltorch_cpu -lc10 -pthread -o main

# Benchmark target
benchmark: $(SRC)
	$(CC) $(CFLAGS) -O3 -march=native -ffast-math $(SRC) $(LDFLAGS) -L/home/jiwokim/torch/libtorch/lib $(LIBS) $(INCLUDES) -ltorch -ltorch_cpu -lc10 -pthread -o main
	./main --benchmark

clean:
	rm -f main *.o