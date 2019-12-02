/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - dartboard.cpp
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
#include <math.h>

using namespace std;
using namespace cv;


/** Function Headers */
void houghSetup(Mat image, Mat &ang, Mat &mag);
void lineMain(Mat &image, Mat &ang, Mat &mag);
void circleMain(Mat &image, Mat &ang, Mat &mag);
//Line Funcs
void conv(cv::Mat &input, Mat kernel, cv::Mat &convOutput);
void grad(Mat &dx, Mat &dy, Mat &mag, Mat &ang);
void houghLine(Mat &mag_thr, Mat &grad_ori, Mat &hspace, int threshold);
void thresholdingLine(Mat &mag, Mat &mag_threshold);
//Circle Funcs
void thresholding(double threshold, Mat &input, Mat &output);
void houghCircle(Mat &mag_thr, Mat &grad_ori, int radius, Mat &hspace);
void houghToMat(Mat hough_array, Mat &output, int radius);

//USED IN LINE FUNCTIONS
int maxDistance; //max diagonal distance of each frame
int precision = 4; //Number we divide 1 degree by

/** @function main */
int main( int argc, const char** argv )
{
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat image = frame.clone();
    Mat out = image.clone();
	Mat mag;
	Mat ang;
	houghSetup(image, ang, mag);

    lineMain(image, ang, mag);
    circleMain(image, ang, mag);

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

void lineMain(Mat &image, Mat &ang, Mat &mag) {
    maxDistance = sqrt(pow(mag.cols,2)+pow(mag.rows,2));  //calculate the max diagonal distance of this frame
    Mat hspaceLine(Size(180*precision, maxDistance*2), CV_64F, Scalar(0));  //generate line hough space container
    std::cout << "hough line thresholding..." << '\n';
    Mat mag_threshold;
    thresholdingLine(mag, mag_threshold);  //magnitude thresholded
    std::cout << "generating hough line..." << '\n';
    houghLine(mag_threshold, ang, hspaceLine, 0);  //generate line hough space

	Mat hough_norm;
	normalize(hspaceLine, hough_norm, 0, 255, NORM_MINMAX);
    std::cout << "-----------------------------" << '\n';
    std::cout << "writing hough_line_thresh.jpg" << '\n';
	imwrite("hough_line_thresh.jpg", mag_threshold);
    std::cout << "writing hough_line.jpg" << '\n';
    std::cout << "-----------------------------" << '\n';
	imwrite("hough_line.jpg", hough_norm);
}

void circleMain(Mat &image, Mat &ang, Mat &mag) {
	Mat output_mag_norm;
	normalize(mag, output_mag_norm, 0, 255, NORM_MINMAX);
	double max;
	minMaxIdx(output_mag_norm, NULL,&max,NULL,NULL);
	Mat output_thresholded;
    std::cout << "hough circle thresholding..." << '\n';
	thresholding(max*0.1, output_mag_norm, output_thresholded);
    std::cout << "generating hough circle..." << '\n';
	int radius = min(ang.rows, ang.cols); //The maximum radius circle that can be found
	int dims[3] = {ang.rows, ang.cols, radius};
	Mat hspace = Mat(3, dims, CV_64F, Scalar(0));
	houghCircle(output_thresholded, ang, radius, hspace); //Have create 3d hough mat

	Mat output_hough;
	output_hough.create(ang.size(), ang.type());
	houghToMat(hspace, output_hough, radius);
	normalize(output_hough, output_hough, 0, 255, NORM_MINMAX);
    std::cout << "-----------------------------" << '\n';
    std::cout << "writing hough_circle_thresh.jpg" << '\n';
	imwrite("hough_circle_thresh.jpg", output_thresholded);
    std::cout << "writing hough_circle.jpg" << '\n';
    std::cout << "-----------------------------" << '\n';
	imwrite("hough_circle.jpg", output_hough);

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
