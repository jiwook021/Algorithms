CC = g++
CFLAGS  = -g -Wall -std=c++23 -Wextra

all: quick_sort_test
main: quick_sort_test

quick_sort_test:
	$(CC) $(CFLAGS) quick_sort_test.cpp -o quick_sort_test -lgtest -lgtest_main -pthread
clean:
	$(RM) quick_sort_test