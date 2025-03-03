CC = g++
CFLAGS  = -g -Wall -std=c++20
Language = cpp

main:
	$(CC) $(CFLAGS) main.$(Language) -o main `pkg-config --cflags --libs libavcodec libavformat libavutil libswscale`
clean:
	$(RM) main
