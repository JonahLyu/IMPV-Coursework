// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

#define rad 60

using namespace cv;
using namespace std;

void GaussianBlur(
	cv::Mat &input, 
	int size,
	cv::Mat &blurredOutput);

void conv(
	cv::Mat &input, 
	int size,
    Mat kernel,
	cv::Mat &convOutput,
	Mat &unNorm);

void grad(
	Mat &dx,
	Mat &dy,
	Mat &mag,
	Mat &ang);

void thresh(
	Mat &mag,
	int thr,
	Mat &out);

void hough(
	Mat &mag,
	Mat &ang,
	Mat &out);

void norm(
	Mat &in);

void hough2Mat(
	Mat hough,
	Mat &out);

void circleHighlight(
	Mat hough,
	int thr,
	Mat &out);

int main( int argc, char** argv )
{

 // LOADING THE IMAGE
 char* imageName = argv[1];

 Mat image;
 image = imread( imageName, 1 );

 if( argc != 2 || !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }

 // CONVERT COLOUR, BLUR AND SAVE
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
// std::cout << "M = " << std::endl << " " << kern << std::endl << std::endl;

Mat blur = gray_image.clone();
// Mat blur;
// GaussianBlur(gray_image,kSize,blur);

 Mat convImdx;
 Mat dxNonNorm;
 conv(blur,kSize,kerndx,convImdx,dxNonNorm);

  Mat convImdy;
  Mat dyNonNorm;
 conv(blur,kSize,kerndy,convImdy,dyNonNorm);

Mat mag;
Mat ang;

grad(dxNonNorm, dyNonNorm, mag, ang);

// Mat test = (convImdy + convImdx)/2;
// Mat test2;
// conv()

 norm(convImdx);
 norm(convImdy);
 norm(mag);

 Mat thr;
 thresh(mag, 30, thr);

	int radius = 50;
	int dims[3] = {ang.cols, ang.rows, radius};
	Mat hspace = Mat(3, dims, CV_64F, 0);

 hough(thr, ang, hspace);
 Mat hMat;
 hMat.create(ang.size(), ang.type());
 hough2Mat(hspace, hMat);
 Mat hl;
 hl.create(ang.size(), ang.type());
 circleHighlight(hspace, 100, hl);
//  norm(hMat);
//  norm(hl);
 norm(ang);
 imwrite( "convdx.jpg", convImdx );
 imwrite( "convdy.jpg", convImdy );
 imwrite( "mag.jpg", mag );
 imwrite( "ang.jpg", ang );
 imwrite( "thr.jpg", thr );
 imwrite( "hMat.jpg", hMat);
 imwrite( "hlight.jpg", hl);
//  imwrite( "test.jpg", test );
// imwrite( "blur.jpg", unsharpMask(gray_image, carBlurred) );

 return 0;
}

int ***malloc3dArray(int dim1, int dim2, int dim3)
{
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
	for (j = 0; j < dim2; j++) {
  	    array[i][j] = (int *) malloc(dim3 * sizeof(int));
	}

    }
    return array;
}

void minMax(int &min, int &max, Mat check) {
	min = check.at<uchar>(0,0);
	max = min;
	for(int x = 0; x < check.cols; x++) {
		for (int y = 0; y < check.rows; y++) {
			if (check.at<double>(y,x) < min) min = check.at<double>(y,x);
			if (check.at<double>(y,x) > max) max = check.at<double>(y,x);
		}
	}
}

void norm(Mat &in) {
	int low;
	int high;
	minMax(low, high, in);
	for (int y = 0; y < in.rows; y++) {
		for (int x = 0; x < in.cols; x++) {
			in.at<double>(y,x) = 255.0 * ((in.at<double>(y,x) - low) / (high - low));
		}
	}
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
	// norm(temp);
	convOutput = temp;
}

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D 
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);
	
	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

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
			blurredOutput.at<uchar>(i, j) = (uchar) sum;
		}
	}
}

void grad(Mat &dx, Mat &dy, Mat &mag, Mat &ang) {
	mag.create(dx.size(), dx.type());
	Mat magOut;
	magOut.create(dx.size(), CV_64F);
	Mat angOut;
	angOut.create(dx.size(), CV_64F);
	for (int y = 0; y < dx.rows; y++) {
		for (int x = 0; x < dx.cols; x++) {
			magOut.at<double>(y, x) = sqrt(pow(dx.at<double>(y,x),2.0) + pow(dy.at<double>(y,x),2.0));
			angOut.at<double>(y, x) = atan2(dy.at<double>(y,x),dx.at<double>(y,x));
			// angOut.at<double>(y, x) = atan(dy.at<double>(y,x)/dx.at<double>(y,x));
			// if (angOut.at<double>(y, x) != 0) cout<<angOut.at<double>(y, x) * (180/M_PI)<<endl;
		}
	}
	// norm(magOut);
	// norm(angOut);
	mag = magOut;
	ang = angOut;
}

void thresh(Mat &mag, int thr, Mat &out) {
	out.create(mag.size(), mag.type());
	for (int y = 0; y < mag.rows; y++) {
		for (int x = 0 ; x < mag.cols; x++) {
			out.at<double>(y, x) = (mag.at<double>(y,x) > thr) ? 255 : 0;
		}
	}
}

void hough(Mat &mag, Mat &ang, Mat &out) {
	double valAng;
	int xs[2];
	int ys[2];
	for (int y = 0; y < mag.rows; y++) {
		for (int x = 0 ; x < mag.cols; x++) {
			for (int r = 0; r < rad; r++) {
				if (mag.at<double>(y,x) > 0) {
					valAng = ang.at<double>(y,x);
					xs[0] = x + (r * cos(valAng));
					xs[1] = x - (r * cos(valAng));
					ys[0] = y + (r * sin(valAng));
					ys[1] = y - (r * sin(valAng));
					for (int i = 0; i < 2; i++) {
						for (int j = 0; j < 2; j++) {
							bool xFlag = (xs[i] > 0 && xs[i] < mag.cols);
							bool yFlag = (ys[j] > 0 && ys[j] < mag.rows);
							if (xFlag && yFlag) out.at<double>(xs[i], ys[i], r)++;
						}
					}
				}
			}
		}
	}
}

void hough2Mat(Mat hough, Mat &out) {
	int rSum;
	for (int y = 0; y < out.rows; y++) {
		for (int x = 0; x < out.cols; x++) {
			rSum = 0;
			for (int r = 0; r < rad; r++) {
				rSum += hough.at<double>(x, y, r);
			} 
			out.at<double>(y,x) = rSum;
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
				if (hough_array.at<long>(x,y,r) > threshold){
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