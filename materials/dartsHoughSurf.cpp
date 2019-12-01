/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

//MENTION THIS ONE AS ATTEMPT TO IMPROVE DETECTION?
//SAY INSTEAD WE USED TECHNIQUES TO APPROXIMATE LOCATION
//OF UNSURE DATBOARDS DUE TO THIS NOT BEING TOATLLY ACCURATE
//AND NOT BEING TOTALLY CONFIDENT IN OUR KNOWLEDGE OF HOW IT WORKED

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <iostream>
#include <string.h>
#include <math.h>

using namespace std;
using namespace cv;

struct circ {
  int x = -1;
  int y = -1;
  int r = -1;
} ;

struct lineS {
	double rho;
	double a;
	double b;
} ;

/** Function Headers */
void houghSetup(Mat image, Mat &ang, Mat &mag);
vector< lineS > lineMain(Mat image, Mat &ang, Mat &mag, Rect pos);
vector<circ> circleMain(Mat &image, Mat &ang, Mat &mag, Rect pos);

vector<Mat> getFrames(Mat image, vector<Rect> det);
bool circRatios(circ circs0, circ circs1);
pair<circ,circ> getCircPair(vector<circ> circs);
vector< lineS > getValidLines(vector< lineS> lines, pair<circ, circ> board, int &count, vector<Point> &iPoints);
bool checkClosePoints(Rect frame, vector<Point> iPoints, int around, int minClose, int validWant, Point &mode);

int pullNum(const char* name);
bool rectIntersect(Rect r1, Rect r2, double thresh);
vector<Rect> getTruths( int index );
vector<Rect> detectAndDisplay( Mat frame, vector<Rect> gt );
void groundTruthDraw(Mat frame, vector<Rect> gt);
vector<Rect> getGT(const char* name);
//Line Funcs
void conv(cv::Mat &input, Mat kernel, cv::Mat &convOutput);
void grad(Mat &dx, Mat &dy, Mat &mag, Mat &ang);
void houghLine(Mat &mag_thr, Mat &grad_ori, Mat &hspace, int threshold);
void suppressLine(Mat &hspace, double bound,int suppRange, Mat &out);
vector< lineS > getLines(Mat &hspace);
Point getIntersect(lineS l1, lineS l2);
bool inCirc(int centreX, int centreY, int radius, Point p1);
void drawLine( Mat &out, double rho, double a, double b, Rect pos, Point center, int scalar);
vector<Point> getAllIntersects(vector< lineS > lines);
void thresholdingLine(Mat &mag, Mat &mag_threshold);
//Circle Funcs
void thresholding(double threshold, Mat &input, Mat &output);
vector< circ > suppressCircles(Mat &hspace, double bound, int cols, int rows, int rad, int suppRange, Mat &out);
void houghCircle(Mat &mag_thr, Mat &grad_ori, int radius, Mat &hspace);
void houghToMat(Mat hough_array, Mat &output, int radius);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

//USED IN LINE FUNCTIONS
int maxDistance; //max diagonal distance of each frame
int precision = 4; //Number we divide 1 degree by

/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat image = frame.clone();
	Mat out = image.clone();
	String s = argv[1];
	String d = "det_" + s;
	s = "out_" + s;

	vector<Rect> gt;

	//Getting correct ground truths for loaded image
	pullNum(argv[1]);
	gt = getGT(argv[1]);
	if (gt.size() == 0) return -1; //Invalid file-name

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	vector<Rect> darts = detectAndDisplay( frame, gt );

	// Draw ground truth rectangles on image
	groundTruthDraw(frame, gt);
	// 4. Save Result Image
	imwrite( d, frame );

	Mat mag;
	Mat ang;
	houghSetup(image, ang, mag);
	Mat output_mag;
	vector<Mat> frames = getFrames(image, darts);
	vector<Mat> framesMag = getFrames(mag, darts);
	vector<Mat> framesAng = getFrames(ang, darts);

    vector<lineS> lines; //Lines found by hough transform
	vector<Point> iPoints; //Intersection points of found lines
	vector<circ> circs; //Circles found by hough transform
	vector<Rect> accepted; //Frames we accept as valid detection
	vector<Rect> potential; //Frames that are potentially accepted, provided a better frame isn't found
	pair<circ,circ> board; //Concentric circles representing a dartboard
	bool dupeFlag; //If the frame we are looking at has already been accepted
	bool pointCheck; //If we have enough points with enough earby neighbours
	Point center; //The point to center our rectangle around

	Mat ref = imread("dart.bmp", CV_LOAD_IMAGE_COLOR);
	Mat grayRef;
	cvtColor( ref, grayRef, CV_BGR2GRAY );

	int minHessian = 400; //Minimum output from hessian filter
	SurfFeatureDetector detector(minHessian); //Creates feature detector
	
	vector<KeyPoint> keyPointRef, keyPointFrame; //points of interest in images
	detector.detect(grayRef, keyPointRef); //Gets points of interest from reference image

	SurfDescriptorExtractor extractor; //Extract descriptors for keypoints using keypoints and image

	Mat descriptorRef, descriptorFrame;

	extractor.compute(grayRef, keyPointRef, descriptorRef);

	BFMatcher matcher(NORM_L2); //Used to match the descriptors generated for reference image for descriptors in checked frame
	vector<DMatch> matches;

	for (int i = 0; i < darts.size(); i++) {
		dupeFlag = false; //Reduces time wasted processing detected region repeatedly
		for (int x = 0; x < accepted.size(); x++) { //Check if detected region has been accepted at already
			if (rectIntersect(darts[i], accepted[x], 0.1)) {
				dupeFlag = true;
				break;
			}
		}
		if (dupeFlag) continue; //Ignore frame if similar frame already accepted
		lines = lineMain(image, framesAng[i], framesMag[i], darts[i]); //Get lines found by hough transform
		if (lines.size() < 5) continue; //Ignore frame if not enough lines present
		
		iPoints = getAllIntersects(lines); //Gets all points of intersection of provided lines

		//Check if number of intersection points close to each other is sufficient
		pointCheck = checkClosePoints(darts[i], iPoints, darts[i].width * 0.1, 10, 5, center);
		if (!pointCheck) continue; //If not ignore frame
		circs = circleMain(image, framesAng[i], framesMag[i], darts[i]); //Get circles in frame
		if (circs.size() < 2) continue; //Ignore frame if not enough circles present
		board = getCircPair(circs); //We assume only one dartboard per frame detected by viola-jones

		if (board.first.r == -1) { //If no valid circle pair discovered
			//Convert current frame to grayscale
			Mat grayImage;
			cvtColor( frames[i], grayImage, CV_BGR2GRAY );
			//Detect keypoints in frame we are looking at
			detector.detect(grayImage, keyPointFrame);
			//Pull out descriptors for these keyPoints
			extractor.compute(grayImage, keyPointFrame, descriptorFrame);
			//Match them to descriptors from the reference dartboard
			matcher.match(descriptorRef, descriptorFrame, matches);
			vector<DMatch> goodMatches;
			//Look through all matches and only keep good matches
			for (DMatch i : matches) {
				for (DMatch j : matches) {
					if (j.distance < 0.75*i.distance) {
						goodMatches.push_back(j);
					}
				}
			}
			//If we have enough good matches accept the current frame
			if (goodMatches.size() > 0.5*keyPointRef.size()) potential.push_back(darts[i]);
			continue;
		}
		
		int count = 0; //count of valid lines
		lines = getValidLines(lines, board, count, iPoints); //Get lines that intersect near the centre of the circle pair
		if (count > 15) { //If we have enough intersections, and valid circle pair
			//Create rect encompassing largest of two concentric circles
			Rect found(board.second.x-board.second.r, board.second.y-board.second.r, 2*board.second.r, 2*board.second.r);
			found.x += darts[i].x;
			found.y += darts[i].y;
			cout << found << endl;
			accepted.push_back(found); //Store found as confident dartboard
			circle(out, Point(board.first.x+darts[i].x, board.first.y+darts[i].y), board.first.r, Scalar(0, 255, 0), 2);
			circle(out, Point(board.second.x+darts[i].x, board.second.y+darts[i].y), board.second.r, Scalar(0, 255, 0), 2);
			//Draw lines
            for (int j = 0; j < lines.size(); j++) {
                drawLine(out, lines[j].rho, lines[j].a, lines[j].b, darts[i], Point(board.first.x, board.first.y), found.width);
            }
			continue;
		}
		//Convert current frame to grayscale
		Mat grayImage;
		cvtColor( frames[i], grayImage, CV_BGR2GRAY );
		//Detect keypoints in frame we are looking at
		detector.detect(grayImage, keyPointFrame);
		//Pull out descriptors for these keyPoints
		extractor.compute(grayImage, keyPointFrame, descriptorFrame);
		//Match them to descriptors from the reference dartboard
		matcher.match(descriptorRef, descriptorFrame, matches);
		vector<DMatch> goodMatches;
		//Look through all matches and only keep good matches
		for (DMatch i : matches) {
			for (DMatch j : matches) {
				if (j.distance < 0.75*i.distance) {
					goodMatches.push_back(j);
				}
			}
		}
		//If we have enough good matches accept the current frame
		if (goodMatches.size() > 0.75*keyPointRef.size()) potential.push_back(darts[i]);
	}

	bool potFlag; //Would check if a area with correct circle ratios but not enough line intersections had been accepted, if not would accept area as dartboard
	for (int i = 0; i < potential.size(); i++) {
		potFlag = false;
		for (int j = 0; j < accepted.size(); j++) {
			if (rectIntersect(potential[i], accepted[j], 0)) { //Have we already stored a better guess for this area
				potFlag = true;
			}
		}
		if (potFlag) { //If better guess found, remove non-confident guess
			potential.erase(potential.begin() + i--);
		} else { //If no better guess found, accept non-confident guess
			cout << potential[i] << endl;
			accepted.push_back(potential[i]);
		}
	}

	for (int i = 0; i < accepted.size(); i++) { //Add accepted frames
		rectangle(out, Point(accepted[i].x, accepted[i].y), Point(accepted[i].x + accepted[i].width, accepted[i].y + accepted[i].height), Scalar( 255, 0, 0 ), 2);
	}
	imwrite(s, out);
    return 0;
}

void houghSetup(Mat image, Mat &ang, Mat &mag) {
	// CONVERT COLOUR AND SAVE
    Mat gray_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );
    int kSize = 3;
    double myKdx[3][3] = {{-1,0,1},
    					{-2,0,2},
    					{-1,0,1}};
    double myKdy[3][3] = {{-1,-2,-1},
    					{0,0,0},
    					{1,2,1}};
    Mat kerndx(3,3, CV_64F, myKdx);
    Mat kerndy(3,3, CV_64F, myKdy);

    Mat convImdx, dxNonNorm;
    conv(gray_image,kerndx,dxNonNorm);

    Mat convImdy, dyNonNorm;
    conv(gray_image,kerndy,dyNonNorm);

    grad(dxNonNorm, dyNonNorm, mag, ang);
}

vector< lineS > lineMain(Mat image, Mat &ang, Mat &mag, Rect pos) {
	//hough line core code

    maxDistance = sqrt(pow(mag.cols,2)+pow(mag.rows,2));  //calculate the max diagonal distance of this frame
    Mat hspaceLine(Size(180*precision, maxDistance*2), CV_64F, Scalar(0));  //generate line hough space container
    Mat mag_threshold;
    thresholdingLine(mag, mag_threshold);  //magnitude thresholded
    houghLine(mag_threshold, ang, hspaceLine, 0);  //generate line hough space

    Mat supHLine;
    suppressLine(hspaceLine, 0.5, 10*precision, supHLine);  //do suppression for the line hough space with 0.5 bound
    vector< lineS > lines;
    lines = getLines(supHLine);  //get all lines of this frame
	return lines;
}

vector<circ> circleMain(Mat &image, Mat &ang, Mat &mag, Rect pos) {
	Mat output_mag_norm;
	normalize(mag, output_mag_norm, 0, 255, NORM_MINMAX);
	double max;
	minMaxIdx(output_mag_norm, NULL,&max,NULL,NULL);
	Mat output_thresholded;
	thresholding(max*0.1, output_mag_norm, output_thresholded);
	int radius = min(ang.rows, ang.cols); //The maximum radius circle that can be found
	int dims[3] = {ang.rows, ang.cols, radius};
	Mat hspace = Mat(3, dims, CV_64F, Scalar(0));
	houghCircle(output_thresholded, ang, radius, hspace); //Have create 3d hough mat
	Mat supH;
	vector< circ > circs;
	circs = suppressCircles(hspace, 0.5, ang.cols, ang.rows, radius, 15, supH); //Suppress 3d hough mat
	return circs;
}

//
//
//

vector<Mat> getFrames(Mat image, vector<Rect> det) {
	vector<Mat> frames;
	for (int i = 0; i < det.size(); i++) {
		if (det[i].x + det[i].width >= image.cols) det[i].width = (image.cols - det[i].x) -1;
		if (det[i].y + det[i].height >= image.rows) det[i].height = (image.rows - det[i].y) -1;
		Mat a = image(det[i]);
		frames.push_back(a);
	}
	return frames;
}

bool circRatios(circ circs0, circ circs1) {
	//circs0 assumed to be inner when checking
	bool yFlag, xFlag, ratioFlag;
	yFlag = (circs0.y > circs1.y-5) && (circs0.y < circs1.y+5);
	xFlag = (circs0.x > circs1.x-5) && (circs0.x < circs1.x+5);
	ratioFlag = (circs1.r <= circs0.r * 2.5) && (circs1.r >= circs0.r * 1.25); //May want to increase 1.25 to 1.5
	if (yFlag && xFlag && ratioFlag) {
		return true;
	}
	return false;
}

//May want to ger pair with closest radius
pair<circ,circ> getCircPair(vector<circ> circs) {
	pair<circ,circ> out;
	for (int i = 0; i < circs.size(); i++) {
		for (int j = 0; j < circs.size(); j++) {
			// if (circRatios(circs[i], circs[j])) out = make_pair(circs[i], circs[j]);
			if (circRatios(circs[i], circs[j])) {
				if (out.first.r == -1) out = make_pair(circs[i], circs[j]); //Prioritses smallest circle pair discovered
				else if (circs[i].r > out.first.r || circs[j].r > out.second.r) { //If smaller circle pair discovered, use them
					out = make_pair(circs[i], circs[j]);
				}
			}
		}
	}
	return out;
}

vector< lineS > getValidLines(vector<lineS> lines, pair<circ, circ> board, int &count, vector<Point> &iPoints) {
	bool added[lines.size()]; //Tracks if line is already tracked as accepted line
	for (int i = 0; i < lines.size(); i++) added[i] = false; //Ensuring all elements start as false
	count = 0;
	vector<lineS> out;
	for (int i = 0; i < lines.size(); i++) {
		for (int j = i+1; j < lines.size(); j++) {
			Point p1 = getIntersect(lines[i],lines[j]);
			if (inCirc(board.first.x, board.first.y, board.first.r * 0.1, p1)) {
				count++;
				iPoints.push_back(p1);
				if (!added[i]) {	//If we aren't already tracking this line, track it
					out.push_back(lines[i]);
					added[i] = true;
				}
				if (!added[j]) {
					out.push_back(lines[j]);
					added[j] = true;
				}
			}
		}
	}
    return out;
}

bool checkClosePoints(Rect frame, vector<Point> iPoints, int around, int minClose, int validWant, Point &mode) {
	bool xFlag, yFlag;
	int countVotes = 0;
	int validPoints = 0;
	int maxPoints = validWant-1;
	for (int i = 0; i < iPoints.size(); i++){
		countVotes = 0;
		for (int j = 0; j < iPoints.size(); j++){
			xFlag = false;
			yFlag = false;
			if (i != j){
				if (iPoints[j].x < (iPoints[i].x + around) && iPoints[j].x > iPoints[i].x - around) xFlag = true;
				if (iPoints[j].y < (iPoints[i].y + around) && iPoints[j].y > iPoints[i].y - around) yFlag = true;
				if (xFlag && yFlag) countVotes++;
			}
		}
		if (countVotes >= minClose) validPoints++;
		if (validPoints > maxPoints) {
			maxPoints = validPoints;
			mode = iPoints[i];
		}
	}	
	return validPoints >= validWant;
}

//
//
//

//Specifc function to pull file number out of files following format xN.jpg where x is a string an N is an integer
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
	if (index < 0 || index > 15) return gt[0];
	return gt[index];
}

void groundTruthDraw(Mat frame, vector<Rect> gt) {
	for (int i = 0; i < gt.size(); i++) {
		rectangle(frame, Point(gt[i].x, gt[i].y), Point(gt[i].x + gt[i].width, gt[i].y + gt[i].height), Scalar( 0, 0, 255 ), 2);
	}
}

/** @function detectAndDisplay */
vector<Rect> detectAndDisplay( Mat frame , vector<Rect> gt)
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
	for( int i = 0; i < darts.size(); i++ ) //Exit loop early when all ground truths seen
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
	float prec = (float) truePos/ darts.size(); //ratio of faces found correctly, to faces detected in image
	float recall = (dartCount > 0 ? (float) truePos/dartCount : 1); //True positive rate
	float f1 = 2 * ((prec * recall) / (prec + recall)); //Measure of accuracy of classifier
	f1 = (f1 != f1) ? 0 : f1; //f1 != f1 is true if f1 is NaN, as long as -ffast-math compiler flag not used
	cout << truePos << " faces out of " << dartCount << " detected correctly." << endl;
	cout << "True positive rate = " <<  recall << endl;
	cout << "F1 score = " << f1 << endl;
	return darts;
}

//////////////////
//LINE FUNCTIONS//
//////////////////

void conv(cv::Mat &input, Mat kernel, cv::Mat &output)
{
	// intialise the output using the input
	output.create(input.size(), DataType<double>::type);

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );


	Mat result(input.rows, input.cols,DataType<double>::type);
	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			output.at<double>(i, j) = sum;
		}
	}

}

bool inCirc(int centreX, int centreY, int radius, Point p1) {
	bool xFlag = (p1.x > centreX - radius) && (p1.x < centreX + radius);
	bool yFlag = (p1.y > centreY - radius) && (p1.y < centreY + radius);
	return xFlag && yFlag;
}

float gradient(pair<Point, Point> line) {
	Point p1 = line.first;
	Point p2 = line.second;
	return (p2.y - p1.y)/(p2.x-p1.x);
}

vector<Point> getAllIntersects(vector< lineS	 > lines){
    vector<Point> iPoints; //Intersection points
	for (int i = 0; i < lines.size(); i++) {
		for (int j = i+1; j < lines.size(); j++) {
			Point p1 = getIntersect(lines[i],lines[j]);
			iPoints.push_back(p1);
		}
	}
    return iPoints;
};

Point getIntersect(lineS l1, lineS l2){
    double c1 = l1.rho;
    double a1 = l1.a;
    double b1 = l1.b;
    double c2 = l2.rho;
    double a2 = l2.a;
    double b2 = l2.b;
    double det = a1*b2 - a2*b1;
    Point p;
    p.x = (b2*c1 - b1*c2) / det;
    p.y = (a1*c2 - a2*c1) / det;
    return p;

}

void drawLine( Mat &out, double rho, double a, double b, Rect pos, Point center, int scalar){
    Point p1,p2,p3,p4;
    double x0 = center.x + pos.x;
    double y0 = (rho-a * center.x)/b+ pos.y;
    int scale = scalar/2;
    p1.x = x0 + scale*(-b);
    p1.y = y0 + scale*a;
    p2.x = x0 - scale*(-b);
    p2.y = y0 - scale*a;
    line(out, p1, p2, Scalar(0,0,255), 2);

}

vector< lineS > getLines(Mat &hspace){
	vector< lineS > lines;
	long max = 0;
	for (int p = 0; p < hspace.rows; p++) {
		for (int theta = 0; theta < hspace.cols; theta++) {
			if (hspace.at<double>(p,theta) > 25){
				lineS l;
                l.rho = p - maxDistance;
                l.a = cos(theta/precision*M_PI/180);
                l.b = sin(theta/precision*M_PI/180);
				lines.push_back(l);
			}
		}
	}
    return lines;

}

void suppressLine(Mat &hspace, double bound,int suppRange, Mat &out) {
	out = Mat(hspace.size(), hspace.type(), Scalar(0));
    Mat tempHspace = hspace.clone();
	if (bound > 1) {
		cout << "bound needs to be 1 or less" << endl;
		return;
	}
	int maxIdx[2];
    double max;
	minMaxIdx(tempHspace, NULL, &max, NULL, maxIdx);
	double loopMax = max;
	while (loopMax > bound * max) {
		out.at<double>(maxIdx[0],maxIdx[1]) = loopMax;
		for (int j = maxIdx[0] - suppRange; j < tempHspace.rows && j < maxIdx[0] + suppRange; j++) {
			for (int i = maxIdx[1] - suppRange; i < tempHspace.cols && i < maxIdx[1] + suppRange; i++) {
				if (j < 0) j = 0;
				if (i < 0) i = 0;
				tempHspace.at<long>(j,i) = 0;
			}
		}
		minMaxIdx(tempHspace, NULL, &loopMax, NULL, maxIdx);
	}
}

void thresholdingLine(Mat &mag, Mat &mag_threshold){
    mag_threshold.create(mag.size(), mag.type());
    double max;
	minMaxIdx(mag, NULL,&max,NULL,NULL);
    double threshold = max*0.5;
    for (int y = 0; y < mag.rows; y++) {
        for (int x = 0; x < mag.cols; x++) {
            if (mag.at<double>(y,x) > threshold){
                mag_threshold.at<double>(y,x) = 255;
            }
            else mag_threshold.at<double>(y,x) = 0;
        }
    }
}

void houghLine(Mat &mag, Mat &ang, Mat &hspace, int threshold){

	for (int y = 0; y < mag.rows; y++) {
		for (int x = 0; x < mag.cols; x++) {
			if (mag.at<double>(y, x) > threshold) {
                for (int theta = 0; theta < 180*precision; theta+=1){
    				double p = x * cos(theta/precision*M_PI/180) + y * sin(theta/precision*M_PI/180);
                    if (p < hspace.rows && theta < hspace.cols){
                        hspace.at<double>(p+maxDistance, theta) += 1;
                    }
                }
			}

		}
	}
}


void grad(Mat &dx, Mat &dy, Mat &mag, Mat &ang) {
	mag.create(dx.size(), dx.type());
    ang.create(dx.size(), dx.type());
	Mat magOut;
	magOut.create(dx.size(), CV_64F);
	Mat angOut;
	angOut.create(dx.size(), CV_64F);
	for (int y = 0; y < dx.rows; y++) {
		for (int x = 0; x < dx.cols; x++) {
			magOut.at<double>(y, x) = sqrt(pow(dx.at<double>(y,x),2.0) + pow(dy.at<double>(y,x),2.0));
			angOut.at<double>(y, x) = atan2(dy.at<double>(y,x),dx.at<double>(y,x));
		}
	}

	mag = magOut;
	ang = angOut;
}

////////////////////
//CIRCLE FUNCTIONS//
////////////////////

void thresholding(double threshold, Mat &input, Mat &output){
	output.create(input.size(),input.type());
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<double>(i,j) > threshold) output.at<double>(i,j) = 255.0;
			else output.at<double>(i,j) = 0.0;
		}
	}
}

vector< circ > suppressCircles(Mat &hspace, double bound, int cols, int rows, int rad, int suppRange, Mat &out) {
	vector< circ > circs;
	int dims[3] = {rows, cols, rad};
	out = Mat(3, dims, CV_64F);
	if (bound > 1) {
		cout << "bound needs to be 1 or less" << endl;
		return circs;
	}
	int maxIdx[3];
	minMaxIdx(hspace, NULL, NULL, NULL, maxIdx);
	double max = hspace.at<double>(maxIdx[0],maxIdx[1],maxIdx[2]);
	double loopMax = max;
	circ c;

	while (loopMax > bound * max) {
		//y,x,r
		c.x = maxIdx[1];
		c.y = maxIdx[0];
		c.r = maxIdx[2];
		circs.push_back(c);
		out.at<double>(maxIdx[0],maxIdx[1],maxIdx[2]) = loopMax;
		for (int j = maxIdx[0] - suppRange; j < rows && j < maxIdx[0] + suppRange; j++) {
			for (int i = maxIdx[1] - suppRange; i < cols && i < maxIdx[1] + suppRange; i++) {
				for (int r = maxIdx[2] - suppRange; r < rad && r < maxIdx[2] + suppRange; r++) {
					if (j < 0) j = 0;
					if (i < 0) i = 0;
					if (r < 0) r = 0;
					hspace.at<long>(j,i,r) = 0;
				}
			}
		}
		minMaxIdx(hspace, NULL, NULL, NULL, maxIdx);
		loopMax = hspace.at<double>(maxIdx[0],maxIdx[1],maxIdx[2]);
	}
	return circs;
}

void houghCircle(Mat &mag_thr, Mat &grad_ori, int radius, Mat &hspace){

	int x0[2], y0[2];
	double valMag, valAng;
	for (int y = 0; y < mag_thr.rows; y++) {
		for (int x = 0; x < mag_thr.cols; x++) {
			for (int r = 0; r < radius; r++) {
				if (mag_thr.at<double>(y, x) > 0) {
					valAng = grad_ori.at<double>(y,x);
					x0[0] = x + (r * cos(valAng));
					x0[1] = x - (r * cos(valAng));
					y0[0] = y + (r * sin(valAng));
					y0[1] = y - (r * sin(valAng));
					for (int m = 0; m < 2; m++) {
						for (int n = 0; n < 2; n++) {
							bool f1 = (y0[n] > 0 && y0[n] < mag_thr.rows);
							bool f2 = (x0[m] > 0 && x0[m] < mag_thr.cols);
							if (f1 && f2) hspace.at<double>(y0[n],x0[m],r) += 1;
						}
					}
				}
			}
		}
	}
}

void houghToMat(Mat hough_array, Mat &output, int radius){
	int rSum;
	for (int y = 0; y < output.rows; y++) {
		for (int x = 0; x < output.cols; x++) {
			rSum = 0;
			for (int r = 0; r < radius; r++) {
				rSum += hough_array.at<double>(y,x,r);
			}
			output.at<double>(y,x) = rSum;
		}
	}

}
