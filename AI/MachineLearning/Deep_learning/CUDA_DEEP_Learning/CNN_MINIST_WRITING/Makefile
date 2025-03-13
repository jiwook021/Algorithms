CC = nvcc
CFLAGS = -O2 -I/usr/include/x86_64-linux-gnu
Language = cu

main:
	$(CC) $(CFLAGS) main.$(Language) -o main -lcudnn -lcublas
clean:
	$(RM) main
	