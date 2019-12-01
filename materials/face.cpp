/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
bool rectIntersect(Rect r1, Rect r2, int thresh);
vector<Rect> getTruths( int index );
void detectAndDisplay( Mat frame, vector<Rect> gt );
void groundTruthDraw(Mat frame, vector<Rect> gt);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	String s = argv[1];
	s = "face_" + s;

	vector<Rect> gTs;

	//Getting correct ground truths for loaded image
	if (!strcmp(argv[1], "dart4.jpg")) gTs = getTruths(0);
	else if (!strcmp(argv[1], "dart5.jpg")) gTs = getTruths(1);
	else if (!strcmp(argv[1], "dart13.jpg")) gTs = getTruths(2);
	else if (!strcmp(argv[1], "dart14.jpg")) gTs = getTruths(3);
	else if (!strcmp(argv[1], "dart15.jpg")) gTs = getTruths(4);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame, gTs );
	// Draw ground truth rectangles on image
	groundTruthDraw(frame, gTs);
	// 4. Save Result Image
	imwrite( s, frame );

	return 0;
}

// Rect intersection(Rect r1, Rect r2) {

// }

// Rect union(Rect r1, Rect r2) {

// }

bool rectIntersect(Rect r1, Rect r2, double thresh) {
	//Returns a boolean indicating if the area
	//shared by both rectangles is greater than the threshold
	return ((float) (r1 & r2).area()/(r1|r2).area() > thresh);
}

vector<Rect> getTruths(int index) {
	vector< vector<Rect> > gTs(5);
	//Adding ground truth location of faces for each image
	//Indexes:
	//	0 = dart4.jpg, 1 = dart5.jpg, 2 = dart13.jpg, 3 = dart14.jpg, 4 = dart15.jpg
	gTs[0].push_back(Rect(342, 107, 147, 147));
	gTs[1].push_back(Rect(513, 177, 55, 55));
	gTs[1].push_back(Rect(641, 184, 59, 59));
	gTs[1].push_back(Rect(191, 214, 65, 65));
	gTs[1].push_back(Rect(290, 242, 63, 63));
	gTs[1].push_back(Rect(425, 231, 68, 68));
	gTs[1].push_back(Rect(58, 249, 64, 64));
	gTs[1].push_back(Rect(554, 244, 69, 69));
	gTs[1].push_back(Rect(673, 246, 64, 64));
	gTs[1].push_back(Rect(60, 135, 63, 63));
	gTs[1].push_back(Rect(377, 190, 57, 57));
	gTs[1].push_back(Rect(250, 164, 57, 57));
	gTs[2].push_back(Rect(412, 123, 122, 122));
	gTs[3].push_back(Rect(461, 216, 103, 103));
	gTs[3].push_back(Rect(726, 189, 100, 100));
	gTs[4].push_back(Rect(360, 110, 100, 100));
	gTs[4].push_back(Rect(548, 121, 54, 54));
	gTs[4].push_back(Rect(65, 123, 86, 86));
	for (int j = 0; j < gTs[index].size(); j++) { //Expanding ground truth rect so detected can be seen
		gTs[index][j].x -= 4;
		gTs[index][j].y -= 4;
		gTs[index][j].width += 8;
		gTs[index][j].height += 8;
	}
	//Maybe we'll add further gTs for dart 15
	return gTs[index];
}

void groundTruthDraw(Mat frame, vector<Rect> gt) {
	for (int i = 0; i < gt.size(); i++) {
		rectangle(frame, Point(gt[i].x, gt[i].y), Point(gt[i].x + gt[i].width, gt[i].y + gt[i].height), Scalar( 0, 0, 255 ), 2);
	}
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame , vector<Rect> gt)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	int truePos = 0;
	int faceCount = gt.size();

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		bool matchFlag = false;
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
		for (int j = 0; j < gt.size(); j++) {
			if (rectIntersect(faces[i], gt[j], 0.75)) {
				gt.erase(gt.begin() + j);
				matchFlag = true;
				break;
			}
		}
		if (matchFlag) {
			cout << "Detected face " << i+1 << " " << faces[i] << " closely matches ground truth" << endl;
			truePos++;
		}
		else cout << "Detected face " << i+1 << " " << faces[i]  << "doesn't match a ground truth face" << endl;
	}
	float precision = (float) truePos/ faces.size(); //ratio of faces found correctly, to faces detected in image
	float recall = (faceCount > 0 ? (float) truePos/faceCount : 1); //True positive rate
	float f1 = 2 * ((precision * recall) / (precision + recall)); //Measure of accuracy of classifier
	f1 = (f1 != f1) ? 0 : f1; //f1 != f1 is true if f1 is NaN, as long as -ffast-math compiler flag not used
	cout << truePos << " faces out of " << faceCount << " detected correctly." << endl;
	cout << "True positive rate = " <<  recall << endl;
	cout << "F1 score = " << f1 << endl;
}
