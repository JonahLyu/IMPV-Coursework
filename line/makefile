.PHONY : line
line = line.cpp
image = dart.bmp

FLAGS = -I/usr/local/opt/opencv@2/include /usr/local/opt/opencv@2/lib/libopencv_core.2.4.dylib /usr/local/opt/opencv@2/lib/libopencv_highgui.2.4.dylib

CC = g++ ${O_LIBS} ${FLAGS} -std=c++11

# %: %.cpp
# 	${CC} ${$@} -o $@
# 	./$@ car1.png

line: ${line}
	${CC} ${line} -o line
	./line ${image}
