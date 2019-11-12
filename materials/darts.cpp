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
	gt = getGT(argv[1]);
	if (gt.size() == 0) return -1; //Invalid file-name

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame, gt );
	// Draw ground truth rectangles on image
	groundTruthDraw(frame, gt);
	// 4. Save Result Image
	cout << frame.cols << " " << frame.rows << endl;
	imwrite( "dart_detected.jpg", frame );

	return 0;
}

// Rect intersection(Rect r1, Rect r2) {

// }

// Rect union(Rect r1, Rect r2) {

// }

vector<Rect> getGT(const char* name) {
	if (!strcmp(name, "dart0.jpg")) return getTruths(0);
	else if (!strcmp(name, "dart1.jpg")) return getTruths(1);
	else if (!strcmp(name, "dart2.jpg")) return getTruths(2);
	else if (!strcmp(name, "dart3.jpg")) return getTruths(3);
	else if (!strcmp(name, "dart4.jpg")) return getTruths(4);
	else if (!strcmp(name, "dart5.jpg")) return getTruths(5);
	else if (!strcmp(name, "dart6.jpg")) return getTruths(6);
	else if (!strcmp(name, "dart7.jpg")) return getTruths(7);
	else if (!strcmp(name, "dart8.jpg")) return getTruths(8);
	else if (!strcmp(name, "dart9.jpg")) return getTruths(9);
	else if (!strcmp(name, "dart10.jpg")) return getTruths(10);
	else if (!strcmp(name, "dart11.jpg")) return getTruths(11);
	else if (!strcmp(name, "dart12.jpg")) return getTruths(12);
	else if (!strcmp(name, "dart13.jpg")) return getTruths(13);
	else if (!strcmp(name, "dart14.jpg")) return getTruths(14);
	else if (!strcmp(name, "dart15.jpg")) return getTruths(15);
	vector<Rect> none(0);
	return none;
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
			// cout << "Detected face " << i+1 << " " << faces[i] << " closely matches ground truth" << endl;
			truePos++;
		}
		// else cout << "Detected face " << i+1 << " " << faces[i]  << "doesn't match a ground truth face" << endl;
	}
	float precision = (float) truePos/ faces.size(); //ratio of faces found correctly, to faces detected in image
	float recall = (faceCount > 0 ? (float) truePos/faceCount : 1); //True positive rate
	float f1 = 2 * ((precision * recall) / (precision + recall)); //Measure of accuracy of classifier
	// cout << truePos << " faces out of " << faceCount << " detected correctly." << endl;
	// cout << "True positive rate = " <<  recall << endl;
	// cout << "F1 score = " << f1 << endl;
}
