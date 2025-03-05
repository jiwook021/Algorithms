CC = g++
CFLAGS  = -g -Wall -std=c++17 \
          -I/home/jiwokim/torch/libtorch/include \
          -I/home/jiwokim/torch/libtorch/include/torch/csrc/api/include \
          -L/home/jiwokim/torch/libtorch/lib \
          -ltorch -lc10 \
          -Wl,-rpath,/home/jiwokim/torch/libtorch/lib
Language = cpp

main:
	$(CC) $(CFLAGS) main.$(Language) -o main 

clean:
	rm -f main
