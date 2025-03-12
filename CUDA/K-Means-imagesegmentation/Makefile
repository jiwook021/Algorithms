CC = nvcc
CFLAGS = -O2
Language = cu
OPENCV_FLAGS = -ccbin g++-10 -I/usr/include/opencv4 -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_features2d

main:
	$(CC) $(CFLAGS) --expt-relaxed-constexpr main.$(Language) -o main $(OPENCV_FLAGS)
clean:
	$(RM) main
	


