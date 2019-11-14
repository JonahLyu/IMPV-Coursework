// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>    //depending on your machine setup

using namespace cv;
using namespace std;

void conv(
	cv::Mat &input,
	Mat kernel,
	cv::Mat &blurredOutput);
void min_Max(Mat &input, double &low, double&high);

long ***suppress(long*** hspace, float bound, int cols, int rows, int rad, int suppRange) {
	if (bound > 1) {
		cout << "bound needs to be 1 or less" << endl;
		return hspace;
	}
	double max;
	int maxIdx[3];
	int dims[3] = {rows, cols, rad};
	Mat hMat = Mat(3, dims, CV_64F, hspace);
	minMaxIdx(hMat, NULL, &max, NULL, maxIdx);
	double loopMax = max;
	while (loopMax > bound * max) {
		for (int j = maxIdx[1] - suppRange; j > 0 && j < rows && j < maxIdx[0] + suppRange; j++) {
			for (int i = maxIdx[0] - suppRange; i > 0 && i < cols && i < maxIdx[1] + suppRange; i++) {
				for (int r = maxIdx[2] - suppRange; r > 0 && r < rad && r < maxIdx[2] + suppRange; r++) {
					hMat.at<long>(j,i,r) = 0;
					cout << "t" << endl;
				}
			}
		}
		// hMat = Mat(3, dims, CV_64F, hspace);
		minMaxIdx(hMat, NULL, &loopMax, NULL, maxIdx);
	}
	return hspace;	
}

long ***malloc3dArray(int dim1, int dim2, int dim3)
{
    int i, j, k;
    long ***array = (long ***) malloc(dim1 * sizeof(long **));

    for (i = 0; i < dim1; i++) {
        array[i] = (long **) malloc(dim2 * sizeof(long *));
		for (j = 0; j < dim2; j++) {
	  	    array[i][j] = (long *) malloc(dim3 * sizeof(long));
			for (k = 0; k < dim3; k++) {
				array[i][j][k] = 0;
			}
		}
    }
    return array;
}


void mag_grad(Mat &dx,Mat &dy, Mat &output){
	output.create(dx.size(),dx.type());
	for (int i = 0; i < dx.rows; i++) {
		for (int j = 0; j < dx.cols; j++) {
			output.at<double>(i,j) = sqrt(pow(dx.at<double>(i,j),2) + pow(dy.at<double>(i,j),2));
		}
	}

}

void angle_grad(Mat &dx,Mat &dy, Mat &output){
	output.create(dx.size(),dx.type());
	for (int i = 0; i < dx.rows; i++) {
		for (int j = 0; j < dx.cols; j++) {
			output.at<double>(i,j) = atan2(dy.at<double>(i,j), dx.at<double>(i,j));
		}
	}
}

void thresholding(double threshold, Mat &input, Mat &output){
	output.create(input.size(),input.type());
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<double>(i,j) > threshold) output.at<double>(i,j) = 255.0;
			else output.at<double>(i,j) = 0.0;
		}
	}
}

long ***hough(Mat &mag_thr, Mat &grad_ori, int radius){
	long ***hough_array;
	hough_array = malloc3dArray(mag_thr.cols,mag_thr.rows,radius);

	int x0[2], y0[2];
	double valMag, valAng;

	for (int x = 0; x < mag_thr.rows; x++) {
		for (int y = 0; y < mag_thr.cols; y++) {
			for (int r = 0; r < radius; r++) {
				if (mag_thr.at<double>(x, y) > 0) {
					valAng = grad_ori.at<double>(x,y);
					y0[0] = y + (r * cos(valAng));
					y0[1] = y - (r * cos(valAng));
					x0[0] = x + (r * sin(valAng));
					x0[1] = x - (r * sin(valAng));
					for (int m = 0; m < 2; m++) {
						for (int n = 0; n < 2; n++) {
							bool f1 = (x0[m] > 0 && x0[m] < mag_thr.rows);
							bool f2 = (y0[n] > 0 && y0[n] < mag_thr.cols);
							if (f1 && f2) hough_array[y0[n]][x0[m]][r]++;
						}
					}

				}
			}
		}
	}

	return hough_array;
}

void houghToMat(long ***hough_array, Mat &output, int radius){
	int rSum;
	for (int x = 0; x < output.rows; x++) {
		for (int y = 0; y < output.cols; y++) {
			rSum = 0;
			for (int r = 0; r < radius; r++) {
				rSum += hough_array[y][x][r];
			}
			output.at<double>(x,y) = rSum;
		}
	}

}

void circleHighlight(long ***hough_array, Mat &output, int threshold, int radius){
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
				if (hough_array[x][y][r] > threshold){
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

 // double myk[3][3] = {{1,1,1},
 //                     {1,1,1},
 //                     {1,1,1}};
 double dx[3][3] = {{-1,0,1},
                     {-1,0,1},
                     {-1,0,1}};
 double dy[3][3] = {{-1,-1,-1},
                     {0,0,0},
                     {1,1,1}};
 cv::Mat kernel_dx = Mat(3, 3, CV_64F, dx);
  cv::Mat kernel_dy = Mat(3, 3, CV_64F, dy);
 // cv::Mat kernel = (Mat_<double>(3,3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);

//  Mat kx = cv::getGaussianKernel(3, -1);
//  Mat ky = cv::getGaussianKernel(3, -1);
//  Mat blur = kx + ky.t();

//  conv(gray_image, blur, gray_image);

 Mat output_dx;
 Mat output_dx_norm;
 conv(gray_image,kernel_dx,output_dx);
 normalize(output_dx, output_dx_norm, 0, 255, NORM_MINMAX);

Mat output_dy;
Mat output_dy_norm;
 conv(gray_image,kernel_dy,output_dy);
 normalize(output_dy, output_dy_norm, 0, 255, NORM_MINMAX);

 Mat output_mag;
 Mat output_mag_norm;
 mag_grad(output_dx,output_dy,output_mag);
 normalize(output_mag, output_mag_norm, 0, 255, NORM_MINMAX);

 Mat output_ang;
 Mat output_ang_norm;
 angle_grad(output_dx,output_dy,output_ang);
 normalize(output_ang, output_ang_norm, 0, 255, NORM_MINMAX);

 imwrite( "output_dx.jpg", output_dx_norm );
 imwrite( "output_dy.jpg", output_dy_norm );
  imwrite( "output_mag.jpg", output_mag_norm );
  imwrite( "output_ang.jpg", output_ang_norm );

//part 2, hough

Mat output_thresholded;
thresholding(40, output_mag_norm, output_thresholded);
imwrite( "output_thresholded.jpg", output_thresholded );

long ***hspace; //Want to change this to mat all the way through
int radius = 50;
hspace = hough(output_thresholded, output_ang, radius); //Have create 3d hough mat
Mat ouput_hough;
ouput_hough.create(output_ang.size(), output_ang.type());
suppress(hspace, 0.8, output_ang.cols, output_ang.rows, radius, 30); //Suppress 3d hough mat
houghToMat(hspace, ouput_hough, radius); //Take 3d hough mat to 2d hough mat

Mat ouput_hough_norm;
normalize(ouput_hough, ouput_hough_norm, 0, 255, NORM_MINMAX);

imwrite( "output_hough.jpg", ouput_hough_norm);

Mat output_circles;
output_circles.create(output_ang.size(), output_ang.type());
circleHighlight(hspace, output_mag_norm, 18, radius);
imwrite( "output_circles.jpg", output_mag_norm);

 return 0;
}


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
