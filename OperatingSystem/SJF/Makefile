CC = g++
CFLAGS  = -g -Wall -std=c++20

.PHONY: all main clean

all: main

main:
	$(CC) $(CFLAGS) main.cpp -o main
test:
	$(CC) $(CFLAGS) test_sjf.cpp -o main_test -lgtest -lgtest_main -pthread
run_test:
	./main_test
clean:
	$(RM) main main_test
	