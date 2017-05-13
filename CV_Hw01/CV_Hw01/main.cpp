#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>

#define IMG_COUNT 6
#define ALPHA 1.0

using namespace std;
using namespace cv;

void generatePlyFile(Mat &depth,int imageWidth,int imageHeight)
{
    ofstream ofs("bunny_.ply", ofstream::out);
    ofs << "ply\nformat ascii 1.0\ncomment alpha="<< ALPHA <<"\nelement vertex "<< imageHeight * imageWidth<<"\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue z\nend_header" << endl;
    
    for(int i = 0 ; i < imageHeight ; i ++)
    {
        for(int j = 0 ; j < imageWidth ; j ++)
        {
            ofs << i << " " << j << " " << depth.at<double>(i,j) << " " << "255 255 255" << endl;
        }
    }
}
void considerXY(Mat &X,Mat &X2,Mat &Y,Mat &Y2, Mat &depthTemp,int imageWidth,int imageHeight)
{
    for(int i = 0 ; i < imageHeight ; i ++)
    {
        for(int j = 0 ; j < imageWidth ; j ++)
        {
            double ratioY = (double)i/(double)imageHeight/2.0;
            double ratioX = (double)j/(double)imageWidth/2.0;
            
            depthTemp.at<double>(i,j) = ratioX * Y2.at<double>(i,j) + (0.5-ratioX) * Y.at<double>(i,j) + ratioY * X2.at<double>(i,j) + (0.5-ratioY) *  X.at<double>(i,j);

        }
    }

}

void considerY(Mat &Y,Mat &Y2, Mat &depthTemp,int imageWidth,int imageHeight)
{
    for(int i = 0 ; i < imageHeight ; i ++)
    {
        for(int j = 0 ; j < imageWidth ; j ++)
        {
            
            double ratioX = (double)j/(double)imageWidth;
            depthTemp.at<double>(i,j) = ratioX * Y2.at<double>(i,j) + (1-ratioX) * Y.at<double>(i,j);
        }
    }
}

void considerX(Mat &X,Mat &X2, Mat &depthTemp,int imageWidth,int imageHeight)
{
    for(int i = 0 ; i < imageHeight ; i ++)
    {
        for(int j = 0 ; j < imageWidth ; j ++)
        {
            double ratioY = (double)i/(double)imageHeight;
            
            
            depthTemp.at<double>(i,j) = ratioY * X2.at<double>(i,j) + (1-ratioY) * X.at<double>(i,j);
            
            
        }
    }
}

void generateDepth(Mat &depth,Mat &dfOfdx,Mat &dfOfdy,int imageWidth,int imageHeight)
{
    Mat depthXFirst(Mat::zeros(imageHeight, imageWidth, CV_64FC(1)));   // 先積X再積Y,上到下 .
    Mat depthX2First(Mat::zeros(imageHeight, imageWidth, CV_64FC(1)));  // 先積X再積Y,下到上 .
    Mat depthYFirst(Mat::zeros(imageHeight, imageWidth, CV_64FC(1)));   // 先積Y再積X,左到右 .
    Mat depthY2First(Mat::zeros(imageHeight, imageWidth, CV_64FC(1)));  // 先積Y再積X,右到左 .
    
    // 先求出depth的 col 1
    for(int i = 1 ; i < imageHeight ; i ++)
    {
        depthYFirst.at<double>(i,0) =  depthYFirst.at<double>(i-1,0) + dfOfdy.at<double>(i-1,0);
    }
    
    // 由左積向右
    for(int i = 0 ; i < imageHeight ; i ++)
    {
        for(int j = 1 ; j < imageWidth ; j ++)
        {
            depthYFirst.at<double>(i,j) = depthYFirst.at<double>(i,j-1) + dfOfdx.at<double>(i,j-1);

        }
    }
    
    // 先求出depth的 col imageWidth-1
    for(int i = 1 ; i < imageHeight ; i ++)
    {
        depthY2First.at<double>(i,imageWidth-1) =  depthY2First.at<double>(i-1,imageWidth-1) + dfOfdy.at<double>(i-1,imageWidth-1);
    }
    
    // 由右積向左
    for(int i = 0 ; i < imageHeight ; i ++)
    {
        for(int j = imageWidth - 2 ; j > -1 ; j --)
        {
            depthY2First.at<double>(i,j) = depthY2First.at<double>(i,j+1) - dfOfdx.at<double>(i,j+1);
            
        }
    }

    // 先求出depth的 row 1
    for(int i = 1 ; i < imageWidth ; i ++)
    {
        depthXFirst.at<double>(0,i) =  depthXFirst.at<double>(0,i-1) + dfOfdx.at<double>(0,i-1);
    }
    
    // 由上積向下 .
    for(int i = 0 ; i < imageWidth ; i ++)
    {
        for(int j = 1 ; j < imageHeight ; j ++)
        {
            depthXFirst.at<double>(j,i) = depthXFirst.at<double>(j-1,i) + dfOfdy.at<double>(j-1,i);
        }
    }
    
    // 先求出depth的 row imageHeight-1
    for(int i = 1 ; i < imageWidth ; i ++)
    {
        depthX2First.at<double>(imageHeight-1,i) =  depthX2First.at<double>(imageHeight-1,i-1) + dfOfdx.at<double>(imageHeight-1,i-1);
    }
    
    // 由下積向上 .
    for(int i = 0 ; i < imageWidth ; i ++)
    {
        for(int j = imageHeight - 2 ; j > -1 ; j --)
        {
            depthX2First.at<double>(j,i) = depthX2First.at<double>(j+1,i) - dfOfdy.at<double>(j+1,i);
        }
    }

    Mat depthTemp(Mat::zeros(imageHeight, imageWidth, CV_64FC(1)));
    
    // 依比例混合 .
    //considerX(depthXFirst,depthX2First,depthTemp,imageWidth,imageHeight); // 方法一
    //considerY(depthYFirst,depthY2First,depthTemp,imageWidth,imageHeight); // 方法二
    considerXY(depthXFirst,depthX2First,depthYFirst,depthY2First, depthTemp,imageWidth,imageHeight); // 方法三
    
    // 使用blur filter : 附近九宮格值平均為該點的值 .
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
            na = (normals.col(i * imageWidth + j)).at<double>(0);
            nb = (normals.col(i * imageWidth + j)).at<double>(1);
            nc = (normals.col(i * imageWidth + j)).at<double>(2);
            
            dfOfdx.at<double>(i,j) = (nc == 0)? (-1.0) * na:(-1.0)*na/nc;
            //dfOfdy.at<double>(i,j) = (nc == 0)? (-1.0*nb):(-1.0)*nb/nc;  // 方法一、二
            dfOfdy.at<double>(i,j) = (nc == 0)? (1.0) * nb:(1.0)*nb/nc;  // 方法三
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
        
        length = sqrt(pow(na,2) + pow(nb,2) + pow(nc,2));
        normals.col(i).at<double>(0,0) = (length == 0)? 0.0:na/length;
        normals.col(i).at<double>(1,0) = (length == 0)? 0.0:nb/length;
        normals.col(i).at<double>(2,0) = (length == 0)? 0.0:nc/length;
    }
}

void calculateNormals(Mat &normals,Mat &Y,Mat &U,int pixelAmount)
{
    
    for(int i = 0 ; i < pixelAmount ; i ++)
    {
        normals.col(i) = (U.t() * U).inv() * U.t() * Y.col(i);
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
    ifstream lightSource("test/special/star/LightSource.txt");
    string line ;
    Mat LightSourceArray[IMG_COUNT];
    int index;
    double x,y,z;
    
    while(getline(lightSource, line))
    {
        if(sscanf(line.c_str(),"pic%d: (%lf,%lf,%lf)",&index,&x,&y,&z) != 4)
            return -1;
        
        LightSourceArray[index-1] = (Mat_<double>(1,3) << x , y , z);
    }
    
    /* Combine to  6 * 3 Matrix (U Matrix) */
    for(int i =0; i < IMG_COUNT; i++)
    {
         LightSourceArray[i].row(0).copyTo(LightSources.row(i));
    }
    
    return 1;
}

void filtImage(Mat *Images,int imageWidth,int imageHeight)
{
    Mat temp;
    for(int k = 0 ; k < IMG_COUNT ; k ++)
    {
        temp = Images[k];
        for(int i = 0 ; i < imageHeight ; i ++)
        {
            for(int j = 0 ; j < imageWidth ; j++)
            {
                vector<uchar> tempVec;
                for(int r = i-1 ; r < i+2 ; r++)
                {
                    for(int s = j-1 ; s < j+2 ; s++)
                    {
                        tempVec.push_back(temp.at<uchar>(r,s));
                    }
                }
                sort(tempVec.begin(), tempVec.end());
                Images[k].at<uchar>(i,j) = tempVec.at(4);
            }
        }
    }
   
}


int loadImage(Mat *Images)
{
    string path ;
    for(int i = 0 ; i < IMG_COUNT ; i ++)
    {
        path = "test/special/star/pic" + to_string(i+1) + ".bmp";
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
    {
        cerr << "Load image failure !" << endl;
        exit(1);
    }
    
    if(loadLightSource(LightSources) == -1)
    {
        cerr << "Load ligt sources failure !" << endl;
        exit(1);
    }
    
    /*  Target : x = (U^TU)^-1 * U^T * y for all pixels
     *  U (IMG# * 3): Light source Matrix .
     *  y (IMG# * 1): pixels intensity . => Y (IMG# * (pixel#))
     *  x (3 * 1) : pixel's normal .    => X (3 * (pixel#))
     */
    
    int imageWidth = Images[0].cols, imageHeight = Images[0].rows;
    int pixelAmount = imageWidth * imageHeight;
    Mat allImagesIntensity(Mat::zeros(IMG_COUNT, pixelAmount, CV_64FC(1)));
    filtImage(Images,imageWidth,imageHeight);
    generateY(imageWidth,imageHeight,pixelAmount,allImagesIntensity,Images);
    
    Mat normals(Mat::zeros(3, pixelAmount,CV_64FC(1)));
    calculateNormals(normals, allImagesIntensity, LightSources, pixelAmount);
    normalizeNormals(normals,pixelAmount);
    
    Mat dfOfdx(Mat::zeros(imageHeight,imageWidth, CV_64FC(1)));
    Mat dfOfdy(Mat::zeros(imageHeight, imageWidth, CV_64FC(1)));
    generateDelta(dfOfdx, dfOfdy, normals, imageWidth, imageHeight);
    
    Mat depth(Mat::zeros(imageHeight, imageWidth, CV_64FC(1)));
    generateDepth(depth, dfOfdx, dfOfdy, imageWidth, imageHeight);

    generatePlyFile(depth, imageWidth, imageHeight);
    
    return 0;
    
}


