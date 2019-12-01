// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>
#include <tuple>

#define rad 60
int maxDistance;

using namespace cv;
using namespace std;

void conv(cv::Mat &input, Mat kernel, cv::Mat &convOutput);
void grad(Mat &dx, Mat &dy, Mat &mag, Mat &ang);
void houghLine(Mat &mag_thr, Mat &grad_ori, Mat &hspace, int threshold);
void suppressLine(Mat &hspace, double bound,int suppRange, Mat &out);
vector< tuple <double, double, double> > lineHighlight(Mat &hspace, Mat &output);
Point getIntersect(tuple <double, double, double> l1, tuple <double, double, double> l2);
bool inCirc(int centreX, int centreY, int radius, Point p1);

void thresholdingLine(Mat &mag, Mat &mag_threshold){
    double max;
	minMaxIdx(mag, NULL,&max,NULL,NULL);
    std::cout << "max: " << max << '\n';
    double bound = 0.5;
    double threshold = max*bound;
    int count = 0;
    for (int y = 0; y < mag.rows; y++) {
        for (int x = 0; x < mag.cols; x++) {
            if (mag.at<double>(y,x) > threshold){
                mag_threshold.at<double>(y,x) = 255;
                count++;
            }
            else mag_threshold.at<double>(y,x) = 0;
        }
    }

    while (count > 0.2 * mag.cols*mag.rows){
        printf("not pass %.3f\n", (float)count/(mag.cols*mag.rows));
        bound += 0.05;
        threshold = max*bound;
        count = 0;
        for (int y = 0; y < mag.rows; y++) {
            for (int x = 0; x < mag.cols; x++) {
                if (mag.at<double>(y,x) > threshold){
                    mag_threshold.at<double>(y,x) = 255;
                    count++;
                }
                else mag_threshold.at<double>(y,x) = 0;
            }
        }
    }
    printf("pass %.3f\n", (float)count/(mag.cols*mag.rows));
    std::cout << mag.cols*mag.rows << '\n';
    std::cout << count << '\n';
}

int main( int argc, char** argv ){
    // LOADING THE IMAGE
    char* imageName = argv[1];

    Mat image;
    image = imread( imageName, 1 );
    // image = image(Rect(154, 72, 66, 146));//12
    // image = image(Rect(325, 151, 66, 68)); //3
    // image = image(Rect(203, 112, 78, 75)); //6
    // image = image(Rect(61, 251, 76, 96)); //8
    // image = image(Rect(195, 38, 248, 248));//9
    // image = image(Rect(912, 146, 41, 75)); //10
    // image = image(Rect(578, 127, 67, 89)); //10
    // image = image(Rect(91, 103, 103, 115)); //10
    // image = image(Rect(161, 85, 95, 101)); //11
    // image = image(Rect(269, 119, 139, 136)); //13



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
    conv(gray_image,kerndx,dxNonNorm);

    Mat convImdy, dyNonNorm;
    conv(gray_image,kerndy,dyNonNorm);

    Mat mag,ang;
    grad(dxNonNorm, dyNonNorm, mag, ang);


    //hough line core code
    maxDistance = sqrt(pow(mag.cols,2)+pow(mag.rows,2));
    Mat hspaceLine(Size(720, maxDistance*2), CV_64F, Scalar(0));
    Mat mag_threshold;
    mag_threshold.create(mag.size(), mag.type());
    thresholdingLine(mag, mag_threshold);
    houghLine(mag_threshold, ang, hspaceLine, 0);

    Mat supHLine;
    suppressLine(hspaceLine, 0.5, 40, supHLine);
    vector< tuple <double, double, double> > lines;
    lines = lineHighlight(supHLine, image);
	vector<Point> iPoints; //Intersection points
	for (int i = 0; i < lines.size(); i++) {
		for (int j = i+1; j < lines.size(); j++) {
			Point p1 = getIntersect(lines[i],lines[j]);
			iPoints.push_back(p1);
    		line(image, p1, Point(p1.x+1, p1.y+1), Scalar(255,0,0), 2);
		}
	}
	// for (Point i : iPoints) {
	// 	if (inCirc(image.cols/2, image.rows/2, 30, i)) {
	// 		cout << i << " in range" << endl;
	// 	} else {
	// 		cout << i << " not in range" << endl;
	// 	}
	// }
    imwrite( "lineDetected.jpg", image);


    //
    normalize(dxNonNorm, convImdx, 0, 255, NORM_MINMAX);
    imwrite( "convdx.jpg", convImdx );
    normalize(dyNonNorm, convImdy, 0, 255, NORM_MINMAX);
    imwrite( "convdy.jpg", convImdy );
    normalize(mag, mag, 0, 255, NORM_MINMAX);
    imwrite( "mag.jpg", mag );
    normalize(ang, ang, 0, 255, NORM_MINMAX);
    imwrite( "ang.jpg", ang );
    normalize(hspaceLine, hspaceLine, 0, 255, NORM_MINMAX);
    imwrite( "hspaceLine.jpg", hspaceLine );
    imwrite( "suppressLine.jpg", supHLine );
    imwrite( "threshold_mag.jpg", mag_threshold );
    return 0;
}

bool inCirc(int centreX, int centreY, int radius, Point p1) {
	bool xFlag = p1.x > centreX - radius && p1.x < centreX + radius;
	bool yFlag = p1.y > centreY - radius && p1.y < centreY + radius;
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
	std::cout << "p\ttheta" << '\n';
	for (int p = 0; p < hspace.rows; p++) {
		for (int theta = 0; theta < hspace.cols; theta++) {
			if (hspace.at<double>(p,theta) > 0){
				std::cout << p <<'\t'<<theta/4<< '\n';
                double rho = p - maxDistance;
                double a = cos(theta/4*M_PI/180);
                double b = sin(theta/4*M_PI/180);
                double x0 = rho * a;
                double y0 = rho * b;
                Point p1,p2;
                p1.x = x0 + 1000*(-b);
                p1.y = y0 + 1000*a;
                p2.x = x0 - 1000*(-b);
                p2.y = y0 - 1000*a;
                line(output, p1, p2, Scalar(0,255,0), 2);
				tuple <double, double, double> line = make_tuple(rho, a, b);
				lines.push_back(line);
			}
		}
	}
    return lines;

}

void suppressLine(Mat &hspace, double bound,int suppRange, Mat &out) {
	out.create(hspace.size(), hspace.type());
    Mat tempHspace = hspace.clone();
	if (bound > 1) {
		cout << "bound needs to be 1 or less" << endl;
		return;
	}
	int maxIdx[2];
    double max;
	minMaxIdx(tempHspace, NULL, &max, NULL, maxIdx);
    // std::cout << max2 << '\n';
	double loopMax = max;
	// // out.at<double>(maxIdx[0],maxIdx[1],maxIdx[2]) = loopMax;
	while (loopMax > bound * max) {
		out.at<double>(maxIdx[0],maxIdx[1]) = loopMax;
        std::cout << "loopmax: "<< loopMax << '\n';
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
                for (int theta = 0; theta < 720; theta+=1){
    				double p = x * cos(theta/4*M_PI/180) + y * sin(theta/4*M_PI/180);
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

void conv(cv::Mat &input, Mat kernel, cv::Mat &output)
{
	// intialise the output using the input
	output.create(input.size(), DataType<double>::type);
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
