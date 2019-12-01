.PHONY : face darts dartsH
fac = face.cpp
dar = darts.cpp
darH = dartsHough.cpp
darHO = dartsHoughOld.cpp
image = dart5.jpg

# FLAGS = -I/usr/local/opt/opencv@2/include /usr/local/opt/opencv@2/lib/libopencv_core.2.4.dylib /usr/local/opt/opencv@2/lib/libopencv_highgui.2.4.dylib

FLAGS = -I/usr/local/opt/opencv@2/include /usr/local/opt/opencv@2/lib/libopencv_core.2.4.dylib /usr/local/opt/opencv@2/lib/libopencv_highgui.2.4.dylib /usr/local/opt/opencv@2/lib/libopencv_nonfree.2.4.dylib

CC = g++ ${O_LIBS} ${FLAGS} -std=c++11

# %: %.cpp
# 	${CC} ${$@} -o $@
# 	./$@ car1.png

face: ${fac}
	${CC} ${fac} -o face
	./face ${image}

darts: ${dar}
	${CC} ${dar} -o darts
	./darts ${image}

dartsHO: ${darHO}
	${CC} ${darHO} -o dartsHoughO
	./dartsHoughO ${image}

dartsH: ${darH}
	${CC} ${darH} -o dartsHough
	./dartsHough ${image}

dartsHS: ${darHS}
	${CC} ${darHS} -o dartsHoughS
	./dartsHoughS ${image}

runAll: ${darH}
	${CC} ${darH} -o dartsHough
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

runAllHS:
	./dartsHoughS dart0.jpg
	./dartsHoughS dart1.jpg
	./dartsHoughS dart2.jpg
	./dartsHoughS dart3.jpg
	./dartsHoughS dart4.jpg
	./dartsHoughS dart5.jpg
	./dartsHoughS dart6.jpg
	./dartsHoughS dart7.jpg
	./dartsHoughS dart8.jpg
	./dartsHoughS dart9.jpg
	./dartsHoughS dart10.jpg
	./dartsHoughS dart11.jpg
	./dartsHoughS dart12.jpg
	./dartsHoughS dart13.jpg
	./dartsHoughS dart14.jpg
	./dartsHoughS dart15.jpg
