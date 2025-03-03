CC = g++
CFLAGS  = -g -Wall -std=c++20
Language = cpp

main:
	$(CC) $(CFLAGS) main.$(Language) -o main `pkg-config --cflags --libs gstreamer-1.0`
clean:
	$(RM) main