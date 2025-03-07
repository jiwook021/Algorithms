CC = g++
CFLAGS = -g -Wall -std=c++17 \
         -I/home/jiwokim/torch/libtorch/include \
         -I/home/jiwokim/torch/libtorch/include/torch/csrc/api/include \
         -D_GLIBCXX_USE_CXX11_ABI=0

LDFLAGS = -L/home/jiwokim/torch/libtorch/lib \
          -ltorch -ltorch_cpu -lc10 \
          -Wl,-rpath,/home/jiwokim/torch/libtorch/lib \
          -pthread

Language = cpp

all: main

main: main.cpp
	$(CC) $(CFLAGS) main.cpp $(LDFLAGS) $(CFLAGS) $(LDFLAGS) -o main $(LDFLAGS)

clean:
	rm -f main *.o
