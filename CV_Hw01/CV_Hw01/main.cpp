#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include <fstream>
#include <stdio.h>

#define IMG_COUNT 6


using namespace std;
using namespace cv;


int loadLightSource(Mat &LightSources)
{
    ifstream lightSource("test/bunny/LightSource.txt");
    string line ;
    Mat LightSourceArray[IMG_COUNT];
    int index;
    float x,y,z;
    
    
    while(getline(lightSource, line))
    {
        if(sscanf(line.c_str(),"pic%d: (%f,%f,%f)",&index,&x,&y,&z) != 4)
            return -1;
        
        cout << index << " " << x << " " << y << " " << z << endl;
        
        LightSourceArray[index-1] = (Mat_<float>(1,3) << x , y , z);
    }
    
    
    /* Combine to  6 * 3 Matrix (U Matrix) */
    for(int i =0; i < IMG_COUNT; i++)
    {
        LightSourceArray[i].row(0).copyTo(LightSources.row(i));
    }
    
    return 1;
}

int loadImage(Mat *Images)
{
    string path ;
    for(int i = 0 ; i < IMG_COUNT ; i ++)
    {
        path = "test/bunny/pic" + to_string(i+1) + ".bmp";
        Images[i] = imread(path, IMREAD_GRAYSCALE);
        if(Images[i].data == NULL)
            return -1;
    }
    return 1;
}

int main(int argc, const char * argv[])
{
    
    Mat Images[IMG_COUNT];
    Mat LightSources(Mat::zeros(IMG_COUNT ,3, CV_64FC(1)));
    
    if(loadImage(Images) == -1)
        cout << "Load image failure !" << endl;
    
    if(loadLightSource(LightSources) == -1)
        cout << "Load ligt sources failure !" << endl;
    
    /*  Target : x = (U^TU)^-1 * U^T * y for all pixels
     *  U (IMG# * 3): Light source Matrix .
     *  y (IMG# * 1): pixels intensity . => Y (IMG# * (pixel#))
     *  x (3 * 1) : pixel's normal .    => X (3 * (pixel#))
     */
    
    int imageWidth = Images[0].cols,imageHeight = Images[0].rows;
    int pixelAmount = imageWidth * imageHeight;
    
    Mat intensity(Mat::zeros(IMG_COUNT, pixelAmount, CV_64FC(1)));
    
    
    
    
    
    /*
     
     
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
    
    */
    
    imshow("CV_window", Images[5]);   /* Params : Windows name / Mat */
    
    waitKey();
    
}


