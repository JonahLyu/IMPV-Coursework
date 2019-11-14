.PHONY : sobel
image=coins1.png

FLAGS = -I/usr/local/opt/opencv@2/include /usr/local/opt/opencv@2/lib/libopencv_core.2.4.dylib /usr/local/opt/opencv@2/lib/libopencv_highgui.2.4.dylib

CC = g++ ${O_LIBS} ${FLAGS}

sobel : sobel.cpp
	$(CC) $< -o $@
	./$@ $(image)
