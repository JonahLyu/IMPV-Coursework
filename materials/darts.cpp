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
#include <string.h>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
int pullNum(const char* name);
bool rectIntersect(Rect r1, Rect r2, int thresh);
vector<Rect> getTruths( int index );
void detectAndDisplay( Mat frame, vector<Rect> gt );
void groundTruthDraw(Mat frame, vector<Rect> gt);
vector<Rect> getGT(const char* name);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	vector<Rect> gt;

	//Getting correct ground truths for loaded image
	pullNum(argv[1]);
	gt = getGT(argv[1]);
	if (gt.size() == 0) return -1; //Invalid file-name

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame, gt );
	// Draw ground truth rectangles on image
	groundTruthDraw(frame, gt);
	// 4. Save Result Image
	// cout << frame.cols << " " << frame.rows << endl;
	imwrite( "dart_detected.jpg", frame );

	return 0;
}

// Rect intersection(Rect r1, Rect r2) {

// }

// Rect union(Rect r1, Rect r2) {

// }

//Speicifc function to pull file number out of files following format xN.jpg where x is a string an N is an integer
int pullNum(const char* name) {
	int num = 0;
	int unit = 1;
	for (int i = strlen(name) - 5; name[i] > 47 && name[i] < 58; i--) {
		num += (name[i] - 48) * unit;
		unit *= 10;
	}
	return num;
}

vector<Rect> getGT(const char* name) {
	// if (!strcmp(name, "dart0.jpg")) return getTruths(0);
	// else if (!strcmp(name, "dart1.jpg")) return getTruths(1);
	// else if (!strcmp(name, "dart2.jpg")) return getTruths(2);
	// else if (!strcmp(name, "dart3.jpg")) return getTruths(3);
	// else if (!strcmp(name, "dart4.jpg")) return getTruths(4);
	// else if (!strcmp(name, "dart5.jpg")) return getTruths(5);
	// else if (!strcmp(name, "dart6.jpg")) return getTruths(6);
	// else if (!strcmp(name, "dart7.jpg")) return getTruths(7);
	// else if (!strcmp(name, "dart8.jpg")) return getTruths(8);
	// else if (!strcmp(name, "dart9.jpg")) return getTruths(9);
	// else if (!strcmp(name, "dart10.jpg")) return getTruths(10);
	// else if (!strcmp(name, "dart11.jpg")) return getTruths(11);
	// else if (!strcmp(name, "dart12.jpg")) return getTruths(12);
	// else if (!strcmp(name, "dart13.jpg")) return getTruths(13);
	// else if (!strcmp(name, "dart14.jpg")) return getTruths(14);
	// else if (!strcmp(name, "dart15.jpg")) return getTruths(15);
	int index = pullNum(name);
	if (index < 0 || index > 16) {
		vector<Rect> none(0);
		return none;
	}
	return getTruths(index);
}

bool rectIntersect(Rect r1, Rect r2, double thresh) {
	//Returns a boolean indicating if the area 
	//shared by both rectangles is greater than the threshold
	return ((float) (r1 & r2).area()/(r1|r2).area() > thresh);
}

vector<Rect> getTruths(int index) {
	vector< vector<Rect> > gt(16);
	//Adding ground truth location of faces for each image
	gt[0].push_back(Rect(452, 17, 150, 179));
	gt[1].push_back(Rect(193, 134, 201, 191));
	gt[2].push_back(Rect(102, 100, 91, 85));
	gt[3].push_back(Rect(325, 151, 66, 68));
	gt[4].push_back(Rect(175, 90, 210, 209));
	gt[5].push_back(Rect(435, 142, 100, 107));
	gt[6].push_back(Rect(203, 112, 78, 75));
	gt[7].push_back(Rect(248, 163, 160, 163));
	gt[8].push_back(Rect(835, 220, 124, 118));
	gt[8].push_back(Rect(61, 251, 76, 96));
	gt[9].push_back(Rect(195, 38, 248, 248));
	gt[10].push_back(Rect(912, 146, 41, 75));
	gt[10].push_back(Rect(578, 127, 67, 89));
	gt[10].push_back(Rect(91, 103, 103, 115));
	gt[11].push_back(Rect(161, 98, 82, 81));
	gt[11].push_back(Rect(436, 111, 53, 77)); //Maybe too hidden to expect detection
	gt[12].push_back(Rect(154, 72, 66, 146));
	gt[13].push_back(Rect(269, 119, 139, 136));
	gt[14].push_back(Rect(114, 98, 139, 132));
	gt[14].push_back(Rect(982, 95, 135, 125));
	gt[15].push_back(Rect(155, 49, 132, 150));
	return gt[index];
}

void groundTruthDraw(Mat frame, vector<Rect> gt) {
	for (int i = 0; i < gt.size(); i++) {
		rectangle(frame, Point(gt[i].x, gt[i].y), Point(gt[i].x + gt[i].width, gt[i].y + gt[i].height), Scalar( 0, 0, 255 ), 2);
	}
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame , vector<Rect> gt)
{
	std::vector<Rect> darts;
	Mat frame_gray;
	int truePos = 0;
	int dartCount = gt.size();

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, darts, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << darts.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < darts.size() && gt.size() > 0; i++ ) //Exit loop early when all ground truths seen
	{
		bool matchFlag = false;
		rectangle(frame, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar( 0, 255, 0 ), 2);
		for (int j = 0; j < gt.size(); j++) {
			if (rectIntersect(darts[i], gt[j], 0.75)) {
				gt.erase(gt.begin() + j);
				matchFlag = true;
				break;
			}
		}
		if (matchFlag) {
			// cout << "Detected face " << i+1 << " " << faces[i] << " closely matches ground truth" << endl;
			truePos++;
		}
		else cout << "Detected face " << i+1 << " " << darts[i]  << "doesn't match a ground truth face" << endl;
	}
	float precision = (float) truePos/ darts.size(); //ratio of faces found correctly, to faces detected in image
	float recall = (dartCount > 0 ? (float) truePos/dartCount : 1); //True positive rate
	float f1 = 2 * ((precision * recall) / (precision + recall)); //Measure of accuracy of classifier
	f1 = (f1 != f1) ? 0 : f1; //f1 != f1 is true if f1 is NaN, as long as -ffast-math compiler flag not used
	cout << truePos << " faces out of " << dartCount << " detected correctly." << endl;
	cout << "True positive rate = " <<  recall << endl;
	cout << "F1 score = " << f1 << endl;
}
