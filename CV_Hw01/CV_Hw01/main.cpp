#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include <fstream>

using namespace std;
using namespace cv;

#define BYTE unsigned char

int main(int argc, const char * argv[])
{
    
    Mat Image = imread("test/bunny/pic1.bmp", IMREAD_GRAYSCALE);
    
    Mat tempImage = Image.clone();
    Mat smallImage(Image.rows / 2, Image.cols / 2, CV_8U);
    
    for (int rowIndex = 0; rowIndex < smallImage.rows; rowIndex++) {
        for (int colIndex = 0; colIndex < smallImage.cols; colIndex++) {
            smallImage.at<uchar>(rowIndex, colIndex) = tempImage.at<uchar>(rowIndex * 2, colIndex * 2);
        }
    }
    
    Mat result(tempImage.rows + smallImage.rows, tempImage.cols, CV_8U, Scalar(0));
    tempImage.copyTo(result(Rect(0, 0, tempImage.cols, tempImage.rows)));
    smallImage.copyTo(result(Rect(0, tempImage.rows, smallImage.cols, smallImage.rows)));
    
    imshow("CV", result);
    waitKey();
    
}
