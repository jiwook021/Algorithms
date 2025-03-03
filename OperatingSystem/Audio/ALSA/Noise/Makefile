CC = g++
CFLAGS  = -g -Wall -std=c++23 -Wextra 
Language = cpp

.PHONY: all main clean

all: main

main:
	$(CC) $(CFLAGS) main.$(Language) -o -lasound -I/usr/include \
    -L/usr/lib \
    -lspeexdsp -lsndfile \
    -o main
clean:
	$(RM) main