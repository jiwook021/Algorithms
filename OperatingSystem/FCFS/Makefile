CC = g++
CC_C = gcc
CFLAGS  = -g -Wall -std=c++20

.PHONY: all main clean

all: main

main:
	$(CC_C) -g -Wall main.c -o main
test:
	$(CC) $(CFLAGS) test_fcfs.cpp -o main_test -lgtest -lgtest_main -pthread
run_test:
	./main_test
clean:
	$(RM) main main_test
	