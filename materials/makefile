.PHONY : face darts dartsH
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

runAll:
	./dartsHough dart0.jpg
	./dartsHough dart1.jpg
	./dartsHough dart2.jpg
	./dartsHough dart3.jpg
	./dartsHough dart4.jpg
	./dartsHough dart5.jpg
	./dartsHough dart6.jpg
	./dartsHough dart7.jpg
	./dartsHough dart8.jpg
	./dartsHough dart9.jpg
	./dartsHough dart10.jpg
	./dartsHough dart11.jpg
	./dartsHough dart12.jpg
	./dartsHough dart13.jpg
	./dartsHough dart14.jpg
	./dartsHough dart15.jpg

