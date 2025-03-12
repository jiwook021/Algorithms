CC = nvcc
CFLAGS = -O2
Language = cu

main:
	$(CC) $(CFLAGS) --expt-relaxed-constexpr main.$(Language) -o main 
clean:
	$(RM) main
	