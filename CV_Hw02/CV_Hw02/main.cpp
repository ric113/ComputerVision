#include <opencv2/highgui/highgui.hpp> // For VS2015
#include <opencv2/xfeatures2d.hpp>


#include <iostream>
#include <ctime>

#define OBJ_NUM 7
#define K 2

using namespace std;
using namespace cv;

void readObjectImg(Mat*);
void readSampleImg(Mat&);
void getFeaturePoints(Ptr<Feature2D>&,vector<KeyPoint>*,Mat*);
void calculateFeatureDescriptor(Ptr<Feature2D>&,vector<KeyPoint>*,Mat*,Mat*);
/* KNN */
void calculateMatches(Mat*,Mat&,vector<KeyPoint>&,vector<Mat_<int>>&);
/* RANSAC */
void ransacProcess(vector<KeyPoint>*,Mat*,vector<KeyPoint>&,Mat&,vector<Mat_<int>>&,Mat*);
vector<int> getRandomSeed(int);
void setUMatrix(Mat&,vector<Point2f>&);
void calculateModelMatrix(Mat&, Mat&);
double calculateInlierRatio(Mat&, Mat_<int>&, vector<KeyPoint>&, vector<KeyPoint>&);
/* Wrapping */
void wrappingForwardProcess(Mat*,Mat*,Mat&);



void DEBUG_showFeature(Mat*,vector<KeyPoint>*,Mat,vector<KeyPoint>);
void DEBUG_showMatches(vector<Mat_<int>>&);

const string PREFIX_PATH = "./test/table/";

void DEBUG_showMatches(vector<Mat_<int>> &matches)
{
    for(int i = 0 ; i < OBJ_NUM ; i ++)
    {
        cout << "Obj img " << i + 1 << " matches size : " << matches[i].size() << endl;
        for(int j = 0 ; j < matches[i].rows ; j ++)
        {
            cout << matches[i].row(j) << endl;
        }
    }
}

void DEBUG_showFeature(Mat *objImage,vector<KeyPoint>* objKeypoints,Mat sampleImg,vector<KeyPoint> sampleKeypoints)
{
    Mat objResult[OBJ_NUM];
    for(int i = 0 ; i < OBJ_NUM ; i ++)
    {
        drawKeypoints(objImage[i], objKeypoints[i], objResult[i], Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }
    
    Mat sampleResult;
    drawKeypoints(sampleImg, sampleKeypoints, sampleResult, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    for(int i = 0 ; i < OBJ_NUM ; i ++)
    {
        imshow(to_string(i), objResult[i]);
        
    }
    
    imshow("sample", sampleResult);
    
}


void wrappingForwardProcess(Mat *homographyMatrix,Mat *objImg,Mat &result)
{
    
    for(int i = 0 ; i < OBJ_NUM ; i ++)
    {
        for(int r = 0 ; r < objImg[i].rows ; r ++)
        {
            for(int c = 0 ; c < objImg[i].cols ; c ++)
            {
                Vec3b RGB = objImg[i].at<Vec3b>(r,c);
                Mat homoOriginPoint = (Mat_<double>(3,1) << c , r, 1);
                if(RGB[0] != 0 && RGB[1] != 0 && RGB[2] != 0)
                {
                    Mat homoWrappedPoint = homographyMatrix[i] * homoOriginPoint;
                    int wrappedX = (int)(homoWrappedPoint.at<double>(0,0) / homoWrappedPoint.at<double>(2,0));
                    int wrappedY = (int)(homoWrappedPoint.at<double>(1,0) / homoWrappedPoint.at<double>(2,0));
                    
                    // cout << wrappedX << "," << wrappedY << endl;
                    if(wrappedX > 0 && wrappedY > 0 &&  wrappedX < result.cols && wrappedY < result.rows)
                        result.at<Vec3b>(wrappedY,wrappedX) = RGB;
                    
                }
            }
        }
    }
    
}

vector<int> getRandomSeed(int range)
{
    vector<int> seeds;
    
    for(int i = 0 ; i < 4 ; i ++)
    {
        
        seeds.push_back(rand() % range);
        for(int j = i-1 ; j >= 0 ; j --)
        {
            if(seeds[i] == seeds[j])
            {
                seeds.pop_back();
                i --;
                break;
            }
        }
    }
    
    return seeds;
    
}

void setUMatrix(Mat &U, vector<Point2f> &pointsAndMatchedPoints)
{
    double x1,x2,x3,x4,y1,y2,y3,y4,X1,X2,X3,X4,Y1,Y2,Y3,Y4 ;
    
    X1 = pointsAndMatchedPoints[0].x;
    X2 = pointsAndMatchedPoints[1].x;
    X3 = pointsAndMatchedPoints[2].x;
    X4 = pointsAndMatchedPoints[3].x;
    Y1 = pointsAndMatchedPoints[0].y;
    Y2 = pointsAndMatchedPoints[1].y;
    Y3 = pointsAndMatchedPoints[2].y;
    Y4 = pointsAndMatchedPoints[3].y;
    
    x1 = pointsAndMatchedPoints[4].x;
    x2 = pointsAndMatchedPoints[5].x;
    x3 = pointsAndMatchedPoints[6].x;
    x4 = pointsAndMatchedPoints[7].x;
    y1 = pointsAndMatchedPoints[4].y;
    y2 = pointsAndMatchedPoints[5].y;
    y3 = pointsAndMatchedPoints[6].y;
    y4 = pointsAndMatchedPoints[7].y;
    
    
    /*
    cout << "Origin " << endl;
    cout << x1 << " " << x2 << " " << x3 << " " << x4 << " " << y1 << " " << y2 << " " << y3 << " " << y4 << endl;
    cout << "Matches " << endl;
    cout << X1 << " " << X2 << " " << X3 << " " << X4 << " " << Y1 << " " << Y2 << " " << Y3 << " " << Y4 << endl;
    */
    
    double temp[] = {X1, Y1, 1,0,0,0,-1*x1*X1,-1*x1*Y1,-1*x1,
        0, 0, 0, X1, Y1, 1, -1*y1*X1, -1*y1*Y1, -1*y1,
        X2, Y2, 1,0,0,0,-1*x2*X2,-1*x2*Y2,-1*x2,
        0, 0, 0, X2, Y2, 1, -1*y2*X2, -1*y2*Y2, -1*y2,
        X3, Y3, 1,0,0,0,-1*x3*X3,-1*x3*Y3,-1*x3,
        0, 0, 0, X3, Y3, 1, -1*y3*X3, -1*y3*Y3, -1*y3,
        X4, Y4, 1,0,0,0,-1*x4*X4,-1*x4*Y4,-1*x4,
        0, 0, 0, X4, Y4, 1, -1*y4*X4, -1*y4*Y4, -1*y4};

    U = Mat(8,9,CV_64F, temp).clone();
    
    // cout << U << endl;

}

void calculateModelMatrix(Mat &modelMatrix, Mat &U)
{
    Mat E = U.t() * U;
    Mat eigenValues;
    Mat eigenVectors;
    
    eigen(E, eigenValues, eigenVectors);
    
    // cout << eigenValues.size() << endl;
    // cout << eigenVectors.size() << endl;
    // cout << E * (eigenVectors.row(8)).t() << endl;
    
    for(int i = 0 ; i < 3 ; i ++)
    {
        for(int j = 0 ; j < 3 ; j ++)
        {
            // cout << eigenVectors.at<double>(minEigenValuePos.y,i * 3 + j) << endl;

            modelMatrix.at<double>(i,j) = eigenVectors.at<double>(8,i * 3 + j);
        }
    }
    
     // cout << modelMatrix << endl;
}

double calculateInlierRatio(Mat &modelMatrix, Mat_<int> &matches, vector<KeyPoint> &sampleKeypoint, vector<KeyPoint> &objectKeypoints)
{
    double inlierRatio = 0.0;
    int inlierCount = 0 ;
    
    for(int i = 0 ; i < objectKeypoints.size() ; i ++)
    {
        Mat objHomoCoordinate = (Mat_<double>(3,1) << objectKeypoints[i].pt.x , objectKeypoints[i].pt.y , 1);
        Mat mappedToSampleHomoCoordinate = modelMatrix * objHomoCoordinate;
        
        // cout << mappedToSampleHomoCoordinate << endl;
        
        Point2f mappedToSamplePoint = Point2f(mappedToSampleHomoCoordinate.at<double>(0,0)/mappedToSampleHomoCoordinate.at<double>(2,0),mappedToSampleHomoCoordinate.at<double>(1,0)/mappedToSampleHomoCoordinate.at<double>(2,0));
        
        
       for(int j = 0 ; j < K ; j++)
        {
            Point2f trueMatchPoint = sampleKeypoint[matches.at<int>(i,j)].pt;
            double dist = sqrt(pow(mappedToSamplePoint.x - trueMatchPoint.x, 2.0) + pow(mappedToSamplePoint.y - trueMatchPoint.y, 2.0));
            
            // cout << "Cal Point : "<< mappedToSamplePoint << endl;
            // cout << "True Point :" << trueMatchPoint << endl;
            
            // cout << dist << endl;
            
            if(dist < 10.0) // Dist < 5 才稱為inlier .
            {
                inlierCount ++;
                break;
            }
        }
        
       
        
        // cout << "DIs :" << dist << endl;
        
       
    }
    
    // cout << inlierCount << "/" << objectKeypoints.size() << endl;
    inlierRatio = (double)inlierCount / (double)objectKeypoints.size();
    
    // cout << "Inlier Ratio : " << inlierRatio << endl;
    return inlierRatio;
}



void ransacProcess(vector<KeyPoint> *objKeypoints,Mat *objDescriptors,vector<KeyPoint> &sampleKeypoints,Mat &sampleDescriptors,vector<Mat_<int>> &matches,Mat *homographyMatrix)
{
    const double RATIO_THRESHOLD = 0.8;
    const int MAX_ITERATE_TIME = 10;        // Test 1 , true value 2000 .
    
    srand ((unsigned)time(NULL));
    
    for(int i = 0 ; i < OBJ_NUM ; i ++)
    {
        double inlierRatio = 0.0;
        double maxInlierRatio = 0.0;
        int iterationCount = 0 ;
        Mat bestMatrix;
        
        cout << "Obj " << i << endl;
        
        
        while(inlierRatio < RATIO_THRESHOLD && iterationCount < MAX_ITERATE_TIME)
        {
            
            
            vector<int> randomPointIndex = getRandomSeed((int)objKeypoints[i].size());
            
            /*
            cout << "Rand :" << endl ;
            cout << randomPointIndex[0] << endl;
            cout << randomPointIndex[1] << endl;
            cout << randomPointIndex[2] << endl;
            cout << randomPointIndex[3] << endl;
            */
            
            // Index - 0 ~ 3 : points , 4 ~ 7 : related points . e.g (0 -> 4) , (1 -> 5) .. etc .
            vector<Point2f> pointsAndMatchedPoints;
            
            for(int k = 0 ; k < 4 ; k ++)
                pointsAndMatchedPoints.push_back((objKeypoints[i])[randomPointIndex[k]].pt);
                
            
            
            for(int j1 = 0 ; j1 < K ; j1 ++)
            {
                for(int j2 = 0 ; j2 < K ; j2 ++)
                {
                    for(int j3 = 0 ; j3 < K ; j3 ++)
                    {
                        for(int j4 = 0 ; j4 < K ; j4 ++)
                        {
                            
                            pointsAndMatchedPoints.push_back(sampleKeypoints[matches[i].at<int>(randomPointIndex[0],j1)].pt);
                            pointsAndMatchedPoints.push_back(sampleKeypoints[matches[i].at<int>(randomPointIndex[1],j2)].pt);
                            pointsAndMatchedPoints.push_back(sampleKeypoints[matches[i].at<int>(randomPointIndex[2],j3)].pt);
                            pointsAndMatchedPoints.push_back(sampleKeypoints[matches[i].at<int>(randomPointIndex[3],j4)].pt);
                            
                            // cout << sampleKeypoints[matches[i].at<int>(randomPointIndex[0],j1)].pt << endl;
                            // cout << sampleKeypoints[matches[i].at<int>(randomPointIndex[1],j2)].pt<< endl;
                            // cout << sampleKeypoints[matches[i].at<int>(randomPointIndex[2],j3)].pt<< endl;
                            // cout << sampleKeypoints[matches[i].at<int>(randomPointIndex[3],j4)].pt<< endl;
                            
                            // calculate H Matrix .
                            Mat U = Mat::zeros(8, 9,CV_64FC(1));
                            setUMatrix(U,pointsAndMatchedPoints);
                            Mat modelMatrix(3,3,CV_64FC(1));
                            calculateModelMatrix(modelMatrix,U);
                            
                            // check inlier ratio .
                            inlierRatio = calculateInlierRatio(modelMatrix, matches[i], sampleKeypoints, objKeypoints[i]);
                            // cout << inlierRatio << endl;
                            
                            // store current best Model .
                            if(inlierRatio > maxInlierRatio)
                            {
                                maxInlierRatio = inlierRatio ;
                                bestMatrix = modelMatrix;
                            }
                            
                            
                            // Refresh .
                            pointsAndMatchedPoints.pop_back();
                            pointsAndMatchedPoints.pop_back();
                            pointsAndMatchedPoints.pop_back();
                            pointsAndMatchedPoints.pop_back();
                        }

                    }

                }
            }
            
     
            
            pointsAndMatchedPoints.clear();
            randomPointIndex.clear();
            iterationCount ++ ;
            
            // cout << "end" << endl;
        }
        cout << maxInlierRatio << endl;
        homographyMatrix[i] = bestMatrix;
    }

    
}

void calculateMatches(Mat *objDescriptors,Mat &sampleDescriptors,vector<KeyPoint> &sampleKeypoints,vector<Mat_<int>> &matches)
{
    

    // cout << sampleKeypoints.size() << endl;
    
    for(int j = 0 ; j < OBJ_NUM ; j ++) // travers all Obj .
    {
        
        // cout << "Obj " << j << endl;
        // cout << objDescriptors[OBJ_NUM].rows << endl;
        Mat_<int> tempMatch = Mat_<int>(objDescriptors[j].rows,K);
        // cout << objDescriptors[j].size << endl;
        for(int k = 0 ; k < objDescriptors[j].rows ; k ++) // traverse all keypoints in Obj(j) .
        {
            double min[K] = {DBL_MAX, DBL_MAX};
                
            for(int i = 0 ; i < sampleKeypoints.size() ; i ++)  // traverse all keypoints in sample .
            {
                
                // cout << sampleDescriptors.row(i).size() << endl;
                // cout << objDescriptors[j].row(k).size() << endl;
                
                
                Mat_<float> tmp(objDescriptors[j].row(k).size());
                double dist;
                absdiff(sampleDescriptors.row(i),objDescriptors[j].row(k),tmp);
                dist = norm(tmp,NORM_L2);
                
                // cout << dist << endl;
                
                for(int l = 0 ; l < K ; l ++)
                {
                    if(dist < min[l])
                    {
                        min[l] = dist;
                        tempMatch.at<int>(k,l) = i ;
                        break;
                    }
                }
                
                // cout << "Obj " << j << endl;
                // cout << "Point : " << k << endl;
                // cout << min[j][0] << " " <<min[j][1] << endl;
            }
                
        }
        // cout << tempMatch << endl;
        matches.push_back(tempMatch);
    }
    
}

void calculateFeatureDescriptor(Ptr<Feature2D> &sift,vector<KeyPoint> *keypoints,Mat *objImg,Mat *descriptors)
{
    for(int i = 0 ; i < OBJ_NUM ; i ++)
    {
        sift->compute(objImg[i], keypoints[i], descriptors[i]);
        // cout << "Object " << i << " des  size :" << descriptors[i].size() << endl;
    }
}

void getFeaturePoints(Ptr<Feature2D> &sift,vector<KeyPoint> *keypoints,Mat *objImg)
{
    for(int i = 0 ; i < OBJ_NUM ; i ++)
    {
        sift->detect(objImg[i], keypoints[i]);
        // cout << "Object " << i << " Key points size :" << keypoints[i].size() << endl;
    }
}

void readSampleImg(Mat &sampleImg)
{
    sampleImg = imread(PREFIX_PATH + "sample.bmp",IMREAD_COLOR);
}

void readObjectImg(Mat *objImg)
{
    string path = "";
    
    for(int i = 0 ; i < OBJ_NUM ; i ++)
    {
        path = PREFIX_PATH + "puzzle"+ to_string(i+1) +".bmp";
        objImg[i] = imread(path.c_str(),IMREAD_COLOR);
        
    }
}



int main() {
    
    // Start time
    time_t startTime = time(NULL);
    
    // Get object images & sample images .
    Mat objImg[OBJ_NUM];
    readObjectImg(objImg);
    Mat sampleImg;
    readSampleImg(sampleImg);
    
    // Create SIFT Obj . feature detector and feature extractor
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
    
    // Feature Detection .
    vector<KeyPoint> objKeypoints[OBJ_NUM];
    getFeaturePoints(sift,objKeypoints,objImg);
    vector<KeyPoint> sampleKeypoints;
    sift->detect(sampleImg, sampleKeypoints);
    
    // Calculate Descriptor .
    Mat objDescriptors[OBJ_NUM];
    calculateFeatureDescriptor(sift, objKeypoints, objImg, objDescriptors);
    Mat sampleDescriptors;
    sift->compute(sampleImg, sampleKeypoints, sampleDescriptors);
    
    // DEBUG_showFeature(objImg, objKeypoints, sampleImg, sampleKeypoints);
    
    // K - NN
    vector<Mat_<int>> matches;  // store the 'index' of keypoints .
    calculateMatches(objDescriptors, sampleDescriptors, sampleKeypoints, matches);
    
    // DEBUG_showMatches(matches);
    
    // RANSAC
    Mat homographyMatrix[OBJ_NUM] ;
    for(int i = 0 ; i < OBJ_NUM ; i ++)
        homographyMatrix[i] = Mat::zeros(3, 3,CV_64FC(1));
    ransacProcess(objKeypoints,objDescriptors,sampleKeypoints,sampleDescriptors,matches,homographyMatrix);
    
    // Wraping
    Mat forwardResult(sampleImg.rows,sampleImg.cols,CV_8UC3,Scalar(0,0,0));
    wrappingForwardProcess(homographyMatrix,objImg,forwardResult);
    
    
    
    // Stick to Target ----
    Mat targetImg = imread(PREFIX_PATH + "target.bmp",IMREAD_COLOR);;
    
    
    // Sift
    vector<KeyPoint> sampleKeypointsArr[1] = {sampleKeypoints};
    vector<KeyPoint> targetKeypoints;
    sift->detect(targetImg, targetKeypoints);
    
    Mat targetDescriptors;
    Mat sampleDescriptorsArr[1] = {sampleDescriptors};
    sift->compute(targetImg, targetKeypoints, targetDescriptors);
    
    
    // K - NN
    vector<Mat_<int>> targetMatches;
    calculateMatches(sampleDescriptorsArr, targetDescriptors, targetKeypoints, targetMatches);
    
    /*
    // RANSAC
    Mat targetHomographyMatrix[1] = {Mat::zeros(3, 3,CV_64FC(1))} ;
    ransacProcess(sampleKeypointsArr,sampleDescriptorsArr,targetKeypoints,targetDescriptors,targetMatches,targetHomographyMatrix);
    
    // Wraping to target .
    Mat result(targetImg.rows,targetImg.cols,CV_8UC3,Scalar(0,0,0));
    Mat homographyMatrixToTarget[OBJ_NUM] ;
    for(int i = 0 ; i < OBJ_NUM ; i ++)
    {
        homographyMatrixToTarget[i] = targetHomographyMatrix[1] * homographyMatrix[i];
    }
    wrappingForwardProcess(homographyMatrixToTarget,objImg,result);
    
    
    imshow("Result", result);
    */
    
    // End time
    time_t endTime = time(NULL);
    cout << "time: " << endTime - startTime << " s" << endl;

    waitKey();
    
    
    return EXIT_SUCCESS;
}

