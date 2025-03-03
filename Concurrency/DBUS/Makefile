CC = g++
CFLAGS  = -g -Wall -std=c++20 -pthread -ldbus-c++-1
Language = cpp

main:
	g++ -g -Wall -std=c++20 -pthread -o calculator_service client.cpp -ldbus-c++-1
clean:
	$(RM) client
	$(RM) server
	$(RM) setup