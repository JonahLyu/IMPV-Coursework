// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

#define rad 60

using namespace cv;
using namespace std;

void conv(cv::Mat &input, int size, Mat kernel, cv::Mat &convOutput, Mat &unNorm);
void grad(Mat &dx, Mat &dy, Mat &mag, Mat &ang);
void houghLine(Mat &mag_thr, Mat &grad_ori, Mat &hspace);

int main( int argc, char** argv ){
    // LOADING THE IMAGE
    char* imageName = argv[1];

    Mat image;
    image = imread( imageName, 1 );

    if( argc != 2 || !image.data ){
        printf( " No image data \n " );
        return -1;
    }
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
    conv(gray_image,kSize,kerndx,convImdx,dxNonNorm);

    Mat convImdy, dyNonNorm;
    conv(gray_image,kSize,kerndy,convImdy,dyNonNorm);

    Mat mag,ang;
    grad(dxNonNorm, dyNonNorm, mag, ang);

    int maxDistance = sqrt(pow(mag.cols,2)+pow(mag.rows,2));
    Mat hspaceLine(Size(180, maxDistance), mag.type(), Scalar(0));
    houghLine(mag, ang, hspaceLine);

    // std::cout << hspaceLine << '\n';

    imwrite( "convdx.jpg", convImdx );
    imwrite( "convdy.jpg", convImdy );
    normalize(mag, mag, 0, 255, NORM_MINMAX);
    imwrite( "mag.jpg", mag );
    normalize(ang, ang, 0, 255, NORM_MINMAX);
    imwrite( "ang.jpg", ang );
    normalize(hspaceLine, hspaceLine, 0, 255, NORM_MINMAX);
    imwrite( "hspaceLine.jpg", hspaceLine );
    return 0;
}

void houghLine(Mat &mag, Mat &ang, Mat &hspace){

	int p;
	double valAng;
    int count = 0;
	for (int y = 0; y < mag.rows; y++) {
		for (int x = 0; x < mag.cols; x++) {
			if (mag.at<double>(y, x) > 100) {
                for (int theta = 0; theta < 180; theta+=1){
    				p = x * cos(theta*M_PI/180) + y * sin(theta*M_PI/180);
                    if (p < hspace.rows && theta < hspace.cols
                        && p >= 0 && theta >= 0){
                            hspace.at<double>(p, theta) += 1;
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

void conv(cv::Mat &input, int size, Mat kernel, cv::Mat &convOutput, Mat &unNorm)
{
	// intialise the output using the input
	Mat temp;
	temp.create(input.size(), CV_64F);
	convOutput.create(input.size(), input.type());

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
			temp.at<double>(i, j) = sum;
		}
	}
	unNorm = temp.clone();
	normalize(temp, convOutput, 0, 255, NORM_MINMAX);
}
