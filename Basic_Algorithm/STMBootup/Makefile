CXX = g++
CC = gcc
CXXFLAGS = -g -Wall -Wextra -std=c++23
CFLAGS = -g -Wall -Wextra

SRCS_CPP = main.cpp bootloader.cpp communication.cpp crc.cpp flashinterface.cpp
SRCS_C = nvic_functions.c stm32f4xx_hal.c stm32f4xx_hal_gpio.c system_stm32f4xx.c
OBJS = $(SRCS_CPP:.cpp=.o) $(SRCS_C:.c=.o)

main: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f main *.o

.PHONY: clean
