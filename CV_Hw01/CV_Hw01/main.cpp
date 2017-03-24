#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include <fstream>
#include <stdio.h>
#include <math.h>

#define IMG_COUNT 6
#define ALPHA 1.0


using namespace std;
using namespace cv;

void generateDepth(Mat &depth,Mat &dfOfdx,Mat &dfOfdy,int imageWidth,int imageHeight)
{
    Mat depthXFirst(Mat::zeros(imageHeight, imageWidth, CV_64FC(1)));
    Mat depthYFirst(Mat::zeros(imageHeight, imageWidth, CV_64FC(1)));
    Mat depthTemp(Mat::zeros(imageHeight, imageWidth, CV_64FC(1)));
    
    
    // 先求出depth的 col 1
    for(int i = 1 ; i < imageHeight ; i ++)
    {
        depthYFirst.at<double>(i,0) = depthYFirst.at<double>(i-1,0) + dfOfdy.at<double>(i-1,0);
    }
    
    // 由左積向右
    for(int i = 0 ; i < imageHeight ; i ++)
    {
        for(int j = 1 ; j < imageWidth ; j ++)
        {
            depthYFirst.at<double>(i,j) = depthYFirst.at<double>(i,j-1) + dfOfdx.at<double>(i,j-1);
        }
    }
    
    // 先求出depth的 row 1
    for(int i = 1 ; i < imageWidth ; i ++)
    {
        depthXFirst.at<double>(0,i) = depthXFirst.at<double>(0,i-1) + dfOfdx.at<double>(0,i-1);
    }
    
    // 由上積向下
    for(int i = 0 ; i < imageWidth ; i ++)
    {
        for(int j = 1 ; j < imageHeight ; j ++)
        {
            depthXFirst.at<double>(j,i) = depthXFirst.at<double>(j-1,i) + dfOfdy.at<double>(j-1,i);
        }
    }
    
    
    // 前面兩個結果平均
    for(int i = 0 ; i < imageHeight ; i ++)
    {
        for(int j = 0 ; j < imageWidth ; j ++)
        {
            depthTemp.at<double>(i,j) = 0 * depthXFirst.at<double>(i,j) + 1 * depthYFirst.at<double>(i,j);
        }
    }
    
    // 使用blur filter 
    for(int i = 1 ; i < imageHeight-1 ; i++)
    {
        for(int j = 1 ; j < imageWidth-1 ; j++)
        {
            double temp = 0;
            for(int k = i-1 ; k < i+2 ; k++)
            {
                for(int s = j-1 ; s < j+2 ; s++)
                {
                    temp += depthTemp.at<double>(k,s);
                }
            }
            depth.at<double>(i,j) = temp/9.0;
        }
    }
    

    
}

void generateDelta(Mat &dfOfdx,Mat &dfOfdy,Mat &normals,int imageWidth,int imageHeight)
{
    double na,nb,nc;
    
    for(int i = 0 ; i < imageHeight ; i ++)
    {
        for(int j = 0 ; j < imageWidth ; j ++)
        {
            na = (normals.col(i * imageWidth + j)).at<double>(0,0);
            nb = (normals.col(i * imageWidth + j)).at<double>(1,0);
            nc = (normals.col(i * imageWidth + j)).at<double>(2,0);
            
            cout << na << " " << nb << " " << nc << endl;
            dfOfdx.at<double>(i,j) = (nc == 0)? 0:-1*na/nc;
            dfOfdy.at<double>(i,j) = (nc == 0)? 0:-1*nb/nc;
        }
    }
}

void normalizeNormals(Mat &normals,int pixelAmout)
{
    double na,nb,nc,length;
    
    for(int i = 0 ; i < pixelAmout ; i ++)
    {
        na = normals.col(i).at<double>(0,0);
        nb = normals.col(i).at<double>(1,0);
        nc = normals.col(i).at<double>(2,0);
        
        //cout << na << " " << nb << " " << nc << endl;
        
        length = sqrt(pow(na,2) + pow(nb,2) + pow(nc,2));
        normals.col(i).at<double>(0,0) = (length == 0)? 0.0:na/length;
        normals.col(i).at<double>(1,0) = (length == 0)? 0.0:nb/length;
        normals.col(i).at<double>(2,0) = (length == 0)? 0.0:nc/length;
        
        //cout << normals.col(i).at<double>(0,0) << " " << normals.col(i).at<double>(1,0) << " " << normals.col(i).at<double>(2,0) << endl;
    }
}

void calculateNormals(Mat &normals,Mat &Y,Mat &U,int pixelAmount)
{
    Mat pseudoInverseOfU = U.inv(CV_SVD);
    
    for(int i = 0 ; i < pixelAmount ; i ++)
    {
        normals.col(i) = pseudoInverseOfU * Y.col(i);
    }
}

void generateY(int imageWidth,int imageHeight,int pixelAmount,Mat &allImagesIntensity,Mat *Images)
{
    for(int i = 0 ; i < IMG_COUNT ; i ++)
    {
        for(int j = 0 ; j < pixelAmount ; j ++)
        {
            allImagesIntensity.at<double>(i,j) = Images[i].at<uchar>(j/imageWidth,j%imageWidth);
        }
    }
}


int loadLightSource(Mat &LightSources)
{
    ifstream lightSource("test/bunny/LightSource.txt");
    string line ;
    Mat LightSourceArray[IMG_COUNT];
    int index;
    double x,y,z;
    
    
    while(getline(lightSource, line))
    {
        if(sscanf(line.c_str(),"pic%d: (%lf,%lf,%lf)",&index,&x,&y,&z) != 4)
            return -1;
        
        cout << index << " " << x << " " << y << " " << z << endl;
        
        LightSourceArray[index-1] = (Mat_<double>(1,3) << x , y , z);
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
    
    int imageWidth = Images[0].cols, imageHeight = Images[0].rows;
    int pixelAmount = imageWidth * imageHeight;
    Mat allImagesIntensity(Mat::zeros(IMG_COUNT, pixelAmount, CV_64FC(1)));
    generateY(imageWidth,imageHeight,pixelAmount,allImagesIntensity,Images);
    
    Mat normals(Mat::zeros(3, pixelAmount,CV_64FC(1)));
    calculateNormals(normals, allImagesIntensity, LightSources, pixelAmount);
    normalizeNormals(normals,pixelAmount);
    
    /* Generate df/dx , df/dy */
    Mat dfOfdx(Mat::zeros(imageHeight,imageWidth, CV_64FC(1)));
    Mat dfOfdy(Mat::zeros(imageHeight, imageWidth, CV_64FC(1)));
    generateDelta(dfOfdx, dfOfdy, normals, imageWidth, imageHeight);
    
    /* Generate Depth */
    Mat depth(Mat::zeros(imageHeight, imageWidth, CV_64FC(1)));
    generateDepth(depth, dfOfdx, dfOfdy, imageWidth, imageHeight);

    ofstream ofs("bunny_surface_XYavg.ply", ofstream::out);
    ofs << "ply\nformat ascii 1.0\ncomment alpha="<< ALPHA <<"\nelement vertex 14400\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue z\nend_header" << endl;
    for(int i = 0 ; i < imageHeight ; i ++)
    {
        for(int j = 0 ; j < imageWidth ; j ++)
        {
            ofs << i << " " << j << " " << depth.at<double>(i,j) << " " << "255 255 255" << endl;
        }
    }
    
    
    
    
    
    
    
    
    
    
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
    
    //imshow("CV_window", Images[5]);   /* Params : Windows name / Mat */
    
    waitKey();
    
}


