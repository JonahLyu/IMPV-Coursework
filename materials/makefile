.PHONY : face darts
fac = face.cpp
dar = darts.cpp
darH = dartsHough.cpp
image = dart5.jpg

FLAGS = -I/usr/local/opt/opencv@2/include /usr/local/opt/opencv@2/lib/libopencv_core.2.4.dylib /usr/local/opt/opencv@2/lib/libopencv_highgui.2.4.dylib

CC = g++ ${O_LIBS} ${FLAGS} -std=c++11

# %: %.cpp
# 	${CC} ${$@} -o $@
# 	./$@ car1.png

face: ${fac}
	${CC} ${fac} -o face

darts: ${dar}
	${CC} ${dar} -o darts
	./darts ${image}

dartsH: ${darH}
	${CC} ${darH} -o dartsHough
	./dartsHough ${image}
