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
#include <tuple>

using namespace std;
using namespace cv;

/** Function Headers */
void houghSetup(Mat image, Mat &ang, Mat &mag);
vector<Point> lineMain(Mat image, Mat &ang, Mat &mag, Rect pos);
vector< tuple<int, int, int> > circleMain(Mat &image, Mat &ang, Mat &mag, Rect pos);

vector<Mat> getFrames(Mat image, vector<Rect> det);
bool circRatios(tuple<int,int,int> circs0, tuple<int,int,int> circs1);

int pullNum(const char* name);
bool rectIntersect(Rect r1, Rect r2, int thresh);
vector<Rect> getTruths( int index );
vector<Rect> detectAndDisplay( Mat frame, vector<Rect> gt );
void groundTruthDraw(Mat frame, vector<Rect> gt);
vector<Rect> getGT(const char* name);
//Line Funcs
void conv(cv::Mat &input, Mat kernel, cv::Mat &convOutput);
void grad(Mat &dx, Mat &dy, Mat &mag, Mat &ang);
void houghLine(Mat &mag_thr, Mat &grad_ori, Mat &hspace, int threshold);
void suppressLine(Mat &hspace, double bound,int suppRange, Mat &out);
vector< tuple <double, double, double> > lineHighlight(Mat &hspace, Mat &output);
Point getIntersect(tuple <double, double, double> l1, tuple <double, double, double> l2);
bool inCirc(int centreX, int centreY, int radius, Point p1);
//Circle Funcs
void thresholding(double threshold, Mat &input, Mat &output);
vector< tuple<int, int, int> > suppressCircles(Mat &hspace, double bound, int cols, int rows, int rad, int suppRange, Mat &out);
void houghCircle(Mat &mag_thr, Mat &grad_ori, int radius, Mat &hspace);
void circleHighlight(Mat hough_array, Mat &output, int threshold, int radius);
void houghToMat(Mat hough_array, Mat &output, int radius);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

int maxDistance;

/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat image = frame.clone();
	Mat out = image.clone();

	vector<Rect> gt;

	//Getting correct ground truths for loaded image
	pullNum(argv[1]);
	gt = getGT(argv[1]);
	if (gt.size() == 0) return -1; //Invalid file-name

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	vector<Rect> darts = detectAndDisplay( frame, gt );
	// vector<Mat> frames = getFrames(image, darts);
	// Draw ground truth rectangles on image
	groundTruthDraw(frame, gt);
	// 4. Save Result Image
	// cout << frame.cols << " " << frame.rows << endl;
	imwrite( "dart_detected.jpg", frame );

	Mat mag;
	Mat ang;
	houghSetup(image, ang, mag);
	vector<Mat> frames = getFrames(image, darts);
	vector<Mat> framesMag = getFrames(mag, darts);
	vector<Mat> framesAng = getFrames(ang, darts);
	vector<Rect> accepted;
	vector<Rect> potential;
	vector< tuple<int, int, int> > circs;
	vector<Point> iPoint;
	// tuple<int, int, int> outer;
	// tuple<int, int, int> inner;
	// vector< tuple< tuple<int, int, int>, tuple<int, int, int> > > pairs;
	bool circCount, circLocs, interLocs;
	int min;
	for (int i = 0; i < frames.size(); i++) {
		circs = circleMain(image, framesAng[i], framesMag[i], darts[i]);
		for (int x = 0; x < circs.size(); x++) {
			for (int y = 0; y < circs.size(); y++) {
				cout << circRatios(circs[x], circs[y]) << endl;
				if (circRatios(circs[x], circs[y])) {
					Rect circleRect = Rect(get<1>(circs[y])-get<2>(circs[y]), get<0>(circs[y])-get<2>(circs[y]), 2*get<2>(circs[y]), 2*get<2>(circs[y]));
					circleRect.x += darts[i].x;
					circleRect.y += darts[i].y;
					potential.push_back(circleRect);
					// pairs.push_back(make_tuple(circs[x], circs[y])); //Pairs of circles that fit what we expect from a dartboard
				}
			}
		}
		// for (int x = 0; x < pairs.size(); x++) {
		// 	Rect circleRect = Rect(get<1>(get<0>(pairs[x]))-get<2>(get<0>(pairs[x])), get<0>(get<0>(pairs[x]))-get<2>(get<0>(pairs[x])), 2*get<2>(get<0>(pairs[x])), 2*get<2>(get<0>(pairs[x])));
		// 	circleRect.x += darts[i].x;
		// 	circleRect.y += darts[i].y;
		// 	potential.push_back(circleRect);
		// }

		framesMag = getFrames(mag, potential);
		framesAng = getFrames(ang, potential);
		for (int x = 0; x < potential.size(); x++) {
			iPoint = lineMain(image, framesAng[i], framesMag[i], potential[i]);
			int count = 0;
			// cout << get<0>(inner) << " " << get<2>(inner) << endl;
			for (Point n : iPoint) {
				// cout << i << endl;
				if (inCirc(potential[x].width/2, potential[x].height/2, potential[x].width/2, n)) {
					count++;
				}
			}
			interLocs = (count > 30) && (count < 60);
			cout << interLocs << " " << darts[i] << endl;
			if (interLocs) accepted.push_back(potential[x]);
		}
		
	}
	// for (int i = 0; i < frames.size(); i++) {
	// 	circs = circleMain(image, framesAng[i], framesMag[i], darts[i]);
	// 	circCount = circs.size() == 2;
	// 	circLocs = circRatios(circs);
	// 	if (circCount && circLocs) {
	// 		min = get<2>(circs[0]);
	// 		inner = circs[0];
	// 		for (int j = 0; j < circs.size(); j++) {
	// 			if (min > get<2>(circs[j])) {
	// 				min = get<2>(circs[j]);
	// 				outer = circs[0];
	// 				inner = circs[j];
	// 			}
	// 		}
	// 		iPoint = lineMain(image, framesAng[i], framesMag[i], darts[i]);
	// 		int count = 0;
	// 		// cout << get<0>(inner) << " " << get<2>(inner) << endl;
	// 		for (Point i : iPoint) {
	// 			// cout << i << endl;
	// 			if (inCirc(get<1>(inner), get<0>(inner), get<2>(inner), i)) {
	// 				count++;
	// 			}
	// 		}
	// 		// int expectedCount = ((1 + iPoint.size()) * iPoint.size())/2;
	// 		// cout << expectedCount << " " << count << endl;
	// 		// cout << count << endl;
	// 		interLocs = (count > 30) && (count < 60);
	// 	}
	// 	// if (circCount && circLocs && interLocs) accepted.push_back(Rect(get<1>(inner)-get<2>(inner),get<0>(inner)-get<2>(inner),2*get<2>(inner),2*get<2>(inner)));
	// 	cout << circCount << " " << circLocs << " " << interLocs << " " << darts[i] << endl;
	// 	if (circCount && circLocs && interLocs) accepted.push_back(darts[i]);
	// }
	for (int i = 0; i < accepted.size(); i++) {
		rectangle(out, Point(accepted[i].x, accepted[i].y), Point(accepted[i].x + accepted[i].width, accepted[i].y + accepted[i].height), Scalar( 255, 0, 0 ), 2);
	}
	imwrite("accepted_frames.jpg", out);
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

vector<Point> lineMain(Mat image, Mat &ang, Mat &mag, Rect pos) {
	//hough line core code

    maxDistance = sqrt(pow(mag.cols,2)+pow(mag.rows,2));
    Mat hspaceLine(Size(180, maxDistance*2), CV_64F, Scalar(0));
	double max;
	minMaxIdx(mag, NULL,&max,NULL,NULL);
    houghLine(mag, ang, hspaceLine, max*0.5);
	// std::cout << maxDistance << '\n';

    Mat supHLine;
    suppressLine(hspaceLine, 0.5, 15, supHLine);
    vector< tuple <double, double, double> > lines;
    lines = lineHighlight(supHLine, image);
	vector<Point> iPoints; //Intersection points
	for (int i = 0; i < lines.size(); i++) {
		for (int j = i+1; j < lines.size(); j++) {
			Point p1 = getIntersect(lines[i],lines[j]);
			iPoints.push_back(p1);
    		line(image, p1, Point(p1.x+1, p1.y+1), Scalar(255,0,0), 3);
		}
	}
	return iPoints;
	// for (Point i : iPoints) {
	// 	if (inCirc(image.cols/2, image.rows/2, 30, i)) {
	// 		cout << i << " in range" << endl;
	// 	} else {
	// 		cout << i << " not in range" << endl;
	// 	}
	// }

    // normalize(hspaceLine, hspaceLine, 0, 255, NORM_MINMAX);
    // imwrite( "hspaceLine.jpg", hspaceLine );
    // imwrite( "suppressLine.jpg", supHLine );
	// imwrite( "lineDetected.jpg", image);
}


vector< tuple<int, int, int> > circleMain(Mat &image, Mat &ang, Mat &mag, Rect pos) {
	Mat output_mag_norm;
	normalize(mag, output_mag_norm, 0, 255, NORM_MINMAX);
	Mat output_thresholded;
	thresholding(40, output_mag_norm, output_thresholded);
	imwrite( "output_thresholded.jpg", output_thresholded );

	// long ***hspace; //Want to change this to mat all the way through
	int radius = 50;
	int dims[3] = {ang.rows, ang.cols, radius};
	Mat hspace = Mat(3, dims, CV_64F, Scalar(0));
	// zeroHspace(hspace, dims);
	houghCircle(output_thresholded, ang, radius, hspace); //Have create 3d hough mat
	Mat output_hough;
	output_hough.create(ang.size(), ang.type());
	Mat supH;
	vector< tuple<int,int,int> > circs;
	circs = suppressCircles(hspace, 0.7, ang.cols, ang.rows, radius, 15, supH); //Suppress 3d hough mat
	return circs;
	// houghToMat(supH, output_hough, radius); //Take 3d hough mat to 2d hough mat
	// Mat output_hough_norm;
	// normalize(output_hough, output_hough_norm, 0, 255, NORM_MINMAX);

	// imwrite( "output_hough.jpg", output_hough_norm);

	// Mat output_circles;
	// output_circles.create(ang.size(), ang.type());
	// circleHighlight(supH, output_mag_norm, 18, radius);
	// for (int x = 0; x < pos.width; x++){
	// 	for (int y = 0; y < pos.height; y++){
	// 		image.at<Vec3b>(y+pos.y,x+pos.x) = output_mag_norm.at<Vec3b>(y,x); 
	// 	}
	// }
	
	// imwrite( "output_circles.jpg", image);
}

//
//
//

vector<Mat> getFrames(Mat image, vector<Rect> det) {
	vector<Mat> frames;
	for (int i = 0; i < det.size(); i++) {
		Mat a = image(det[i]);
		frames.push_back(a);
	}
	return frames;
}

bool circRatios(tuple<int,int,int> circs0, tuple<int,int,int> circs1) {
	//circs0 assumed to be inner when checking
	bool yFlag, xFlag, ratioFlag;
	yFlag = (get<0>(circs0) > get<0>(circs1)-5) && (get<0>(circs0) < get<0>(circs1)+5);
	xFlag = (get<1>(circs0) > get<1>(circs1)-5) && (get<1>(circs0) < get<1>(circs1)+5);
	ratioFlag = (get<2>(circs1) <= get<2>(circs0) * 2.5) && (get<2>(circs1) >= get<2>(circs0) * 1.25);
	if (yFlag && xFlag && ratioFlag) {
		return true;
	}
	return false;
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
	float precision = (float) truePos/ darts.size(); //ratio of faces found correctly, to faces detected in image
	float recall = (dartCount > 0 ? (float) truePos/dartCount : 1); //True positive rate
	float f1 = 2 * ((precision * recall) / (precision + recall)); //Measure of accuracy of classifier
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

	// create the Gaussian kernel in 1D
	// cv::Mat kX = cv::getGaussianKernel(size, -1);
	// cv::Mat kY = cv::getGaussianKernel(size, -1);
	//
	// // make it 2D multiply one by the transpose of the other
	// cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

       // SET KERNEL VALUES
	// for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
	//   for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
    //        kernel.at<double>(m+ kernelRadiusX, n+ kernelRadiusY) = (double) 1.0/(size*size);
	//
    //    }

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

Point getIntersect(tuple <double, double, double> l1, tuple <double, double, double> l2){
    double c1 = get<0>(l1);
    double a1 = get<1>(l1);
    double b1 = get<2>(l1);
    double c2 = get<0>(l2);
    double a2 = get<1>(l2);
    double b2 = get<2>(l2);
    double det = a1*b2 - a2*b1;
    Point p;
    p.x = (b2*c1 - b1*c2) / det;
    p.y = (a1*c2 - a2*c1) / det;
    return p;

}

vector< tuple <double, double, double> > lineHighlight(Mat &hspace, Mat &output){
	vector< tuple <double, double, double> > lines;
	long max = 0;
	// std::cout << "p\ttheta" << '\n';
	for (int p = 0; p < hspace.rows; p++) {
		for (int theta = 0; theta < hspace.cols; theta++) {
			if (hspace.at<double>(p,theta) > 0){
				// std::cout << p <<'\t'<<theta<< '\n';
                double rho = p - maxDistance;
                double a = cos(theta*M_PI/180);
                double b = sin(theta*M_PI/180);
                double x0 = rho * a;
                double y0 = rho * b;
                Point p1,p2;
                p1.x = x0 + 1000*(-b);
                p1.y = y0 + 1000*a;
                p2.x = x0 - 1000*(-b);
                p2.y = y0 - 1000*a;
                line(output, p1, p2, Scalar(0,255,0), 3);
				tuple <double, double, double> line = make_tuple(rho, a, b);
				lines.push_back(line);
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
	// std::cout <<  << '\n';
	double loopMax = max;
	// out.at<double>(maxIdx[0],maxIdx[1],maxIdx[2]) = loopMax;
	while (loopMax > bound * max) {
		out.at<double>(maxIdx[0],maxIdx[1]) = loopMax;
		for (int j = maxIdx[0] - suppRange; j < tempHspace.rows && j < maxIdx[0] + suppRange; j++) {
			for (int i = maxIdx[1] - suppRange; i < tempHspace.cols && i < maxIdx[1] + suppRange; i++) {
				if (j < 0) j = 0;
				if (i < 0) i = 0;
				tempHspace.at<long>(j,i) = 0;
			}
		}
	// 	// hMat = Mat(3, dims, CV_64F, hspace);
		minMaxIdx(tempHspace, NULL, &loopMax, NULL, maxIdx);
	// 	loopMax = hspace.at<double>(maxIdx[0],maxIdx[1],maxIdx[2]);
	}
}

void houghLine(Mat &mag, Mat &ang, Mat &hspace, int threshold){

	for (int y = 0; y < mag.rows; y++) {
		for (int x = 0; x < mag.cols; x++) {
			if (mag.at<double>(y, x) > threshold) {
                for (int theta = 0; theta < 180; theta+=1){
    				double p = x * cos(theta*M_PI/180) + y * sin(theta*M_PI/180);
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
	// norm(magOut);
	// norm(angOut);
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

vector< tuple<int, int, int> > suppressCircles(Mat &hspace, double bound, int cols, int rows, int rad, int suppRange, Mat &out) {
	vector< tuple<int, int, int> > circs;
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
	// out.at<double>(maxIdx[0],maxIdx[1],maxIdx[2]) = loopMax;
	while (loopMax > bound * max) {
		//y,x,r
		circs.push_back(make_tuple(maxIdx[0],maxIdx[1],maxIdx[2]));
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
		// hMat = Mat(3, dims, CV_64F, hspace);
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

void circleHighlight(Mat hough_array, Mat &output, int threshold, int radius){
	int x0,y0;
	long max = 0;
	std::cout << "x\ty\tr" << '\n';
	for (int y = 0; y < output.rows; y++) {
		for (int x = 0; x < output.cols; x++) {
			for (int r = 0; r < radius; r++) {
				// if (max < hough_array[x][y][r] ){
				// 	max =  hough_array[x][y][r] ;
				// 	std::cout << "max:"<< max << '\n';
				// }
				if (hough_array.at<long>(y,x,r) > threshold){
					std::cout << x <<'\t'<<y<<'\t'<<r << '\n';
					for (int d = 0; d < 360; d += 1) {
						x0 = x + r * cos(d);
						y0 = y + r * sin(d);
						if (x0 < output.cols && y0 < output.rows) output.at<double>(y0,x0) = 255.0;
					}
				}
			}
		}
	}
}
