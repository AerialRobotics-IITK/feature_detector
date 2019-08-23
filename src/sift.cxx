#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <opencv-3.3.1-dev/opencv2/highgui/highgui.hpp>
#include <opencv-3.3.1-dev/opencv2/features2d/features2d.hpp>
#include <opencv-3.3.1-dev/opencv2/core/core.hpp>
#include <opencv-3.3.1-dev/opencv2/imgproc/imgproc.hpp>
#include <opencv-3.3.1-dev/opencv2/objdetect/objdetect.hpp>
#include <opencv-3.3.1-dev/opencv2/xfeatures2d.hpp>
#include <opencv-3.3.1-dev/opencv2/opencv.hpp>
#include <opencv-3.3.1-dev/opencv2/opencv_modules.hpp>

// #include "opencv2/contrib/contrib.hpp"/

#include <iostream>
#include <fstream>
#define check(X) cout << X << endl

using namespace cv;
using namespace std;

const string defaultDetectorType = "SIFT";
// const string defaultDescriptorType = "SIFT";
const string defaultMatcherType = "FlannBased";
const string defaultQueryImageName = "in/scene.jpg";
const string defaultFileWithTrainImages = "in/train.txt";
const string defaultDirToSaveResImages = "out";

int sceneID = 0, numTrainImages = 0;
bool debug = true, stream = true;
cv::Mat queryImage, markedImage;

static void imageCb(const sensor_msgs::Image msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);  
    }
    catch(cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    markedImage = cv_ptr->image;
    cvtColor(markedImage, queryImage, COLOR_BGR2GRAY);
    sceneID = msg.header.seq;
    return;
}

// static void printPrompt( const string& applName )
// {
//   ROS_INFO("/*\n * This is a sample on matching descriptors detected on one image to descriptors detected in image set.\n
                    // * So we have one query image and several train images. For each keypoint descriptor of query image\n
                    // * the one nearest train descriptor is found the entire collection of train images. To visualize the result\n
                    // * of matching we save images, each of which combines query and train image with matches between them (if they exist).\n
                    // * Match is drawn as line between corresponding points. Count of all matches is equel to count of\n
                    // * query keypoints, so we have the same count of lines in all set of result images (but not for each result\n
                    // * (train) image).\n
                    // */\n\n");

//   ROS_INFO("Format:\n\n");
//   ROS_INFO("./%s [detectorType] [descriptorType] [matcherType] [queryImage] [fileWithTrainImages] [dirToSaveResImages]\n", applName);

//   ROS_INFO("\nExample:\n ./%s %s %s %s %s %s %s", applName, defaultDetectorType, defaultDescriptorType, defaultMatcherType, 
                                                // defaultQueryImageName, defaultFileWithTrainImages, defaultDirToSaveResImages )
// }

static void maskMatchesByTrainImgIdx( const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask )
{
    mask.resize( matches.size() );
    fill( mask.begin(), mask.end(), 0 );
    for( size_t i = 0; i < matches.size(); i++ )
    {
        if( matches[i].imgIdx == trainImgIdx )
            mask[i] = 1;
    }
}

static void readTrainFilenames( const string& filename, string& dirName, vector<string>& trainFilenames )
{
    trainFilenames.clear();

    ifstream file( filename.c_str() );
    if ( !file.is_open() )
        return;

    size_t pos = filename.rfind('\\');
    char dlmtr = '\\';
    if (pos == String::npos)
    {
        pos = filename.rfind('/');
        dlmtr = '/';
    }
    dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;

    while( !file.eof() )
    {
        string str; getline( file, str );
        if( str.empty() ) break;
        trainFilenames.push_back(str);
    }
    file.close();
}

static bool createDetectorDescriptorMatcher( 
                                      const string& matcherType,
                                      Ptr<Feature2D>& detector,
                                      Ptr<DescriptorMatcher>& descriptorMatcher )
{
    // ROS_INFO("< Creating feature detector, descriptor extractor and descriptor matcher ...\n");
    // featureDetector = FeatureDetector::create( detectorType );
    // descriptorExtractor = DescriptorExtractor::create( descriptorType );
    detector = xfeatures2d::SIFT::create();
    descriptorMatcher = DescriptorMatcher::create( matcherType );

    bool isCreated = !( detector.empty() || descriptorMatcher.empty() );
    if( !isCreated )    ROS_ERROR("Can not create feature detector or descriptor extractor or descriptor matcher of given types.\n");

    return isCreated;
}

static bool readTrainImages( const string& trainFilename, vector <Mat>& trainImages, vector<string>& trainImageNames )
{
    // ROS_INFO("< Reading the images...\n");
    // queryImage = imread( queryImageName, CV_LOAD_IMAGE_GRAYSCALE);
    // if( queryImage.empty() )
    // {
        // ROS_ERROR("Query image can not be read. \n");
        // return false;
    // }
    string trainDirName;
    readTrainFilenames( trainFilename, trainDirName, trainImageNames );
    if( trainImageNames.empty() )
    {
        ROS_ERROR("Train image filenames can not be read.>\n");
        return false;
    }
    int readImageCount = 0;
    for( size_t i = 0; i < trainImageNames.size(); i++ )
    {
        string filename = trainDirName + trainImageNames[i];
        Mat img = imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
        if( img.empty() ) ROS_ERROR("Train image can not be read.\n");
        else readImageCount++;
        trainImages.push_back( img );
    }
    if( !readImageCount )
    {
        ROS_ERROR("All train images can not be read.>\n");
        return false;
    }
    // else ROS_INFO("%d train images were read.\n", readImageCount);
    return true;
}

static void detectAndComputeTrainKeypoints( const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints, 
                                    vector<Mat>& trainDescriptors, Ptr<Feature2D>& detector )
{
    //   ROS_INFO("< Computing descriptors for training images...\n");
    detector->detect( trainImages, trainKeypoints );
    detector->compute( trainImages, trainKeypoints, trainDescriptors); 
    int totalTrainDesc = 0;
    if(debug)
    {
        for( vector<Mat>::const_iterator tdIter = trainDescriptors.begin(); tdIter != trainDescriptors.end(); tdIter++ )
            totalTrainDesc += tdIter->rows;
        ROS_INFO("Total train descriptors count: %d", totalTrainDesc );
    }
}

static void detectAndComputeQueryKeypoints( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                    Mat& queryDescriptors, Ptr<Feature2D> detector)
{
    //   ROS_INFO("< Computing descriptors for query image...\n");
    detector->detectAndCompute( queryImage, noArray(), queryKeypoints, queryDescriptors );
    if(debug) ROS_INFO("Query descriptors count: %d", queryDescriptors.rows );
}

// static void detectKeypoints( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
//                       const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints,
//                       Ptr<FeatureDetector>& featureDetector )
// {
    // ROS_INFO("< Extracting keypoints from images...\n");
//     featureDetector->detect( queryImage, queryKeypoints );
    // featureDetector->detect( trainImages, trainKeypoints );
// }

// static void computeDescriptors( const Mat& queryImage, vector<KeyPoint>& queryKeypoints, Mat& queryDescriptors,
//                          const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints, vector<Mat>& trainDescriptors,
//                          Ptr<DescriptorExtractor>& descriptorExtractor )
// {
//     ROS_INFO("< Computing descriptors for keypoints...\n");
//     descriptorExtractor->compute( queryImage, queryKeypoints, queryDescriptors );
//     descriptorExtractor->compute( trainImages, trainKeypoints, trainDescriptors );

//     int totalTrainDesc = 0;
//     for( vector<Mat>::const_iterator tdIter = trainDescriptors.begin(); tdIter != trainDescriptors.end(); tdIter++ )
//         totalTrainDesc += tdIter->rows;

//      ROS_INFO("Query descriptors count: %d; Total train descriptors count: %d", queryDescriptors.rows, totalTrainDesc );
// }

static void matchDescriptors( const Mat& queryDescriptors, const vector<Mat>& trainDescriptors,
                       vector<DMatch>& matches, Ptr<DescriptorMatcher>& descriptorMatcher )
{
    // ROS_INFO("< Set train descriptors collection in the matcher and match query descriptors to them...\n");
    TickMeter tm;
    vector<vector<DMatch>> nn_matches;

    tm.start();
    descriptorMatcher->add( trainDescriptors );
    descriptorMatcher->train();
    tm.stop();
    double buildTime = tm.getTimeMilli();

    tm.start();
    descriptorMatcher->knnMatch( queryDescriptors, nn_matches, 2);
    tm.stop();
    double matchTime = tm.getTimeMilli();

    for(int i=0; i<nn_matches.size(); i++){
        if(nn_matches[i][0].distance < nn_matches[i][1].distance){
            matches.push_back(nn_matches[i][0]);
        }
    }

    CV_Assert( queryDescriptors.rows >= (int)matches.size() || matches.empty() );

    // ROS_INFO("Number of matches: %d", matches.size() );
    // ROS_INFO("Build time: %lf ms; Match time: %lf ms\n", buildTime, matchTime);

    // ROS_INFO("%d\n", queryDescriptors.rows);
    // for(int i=0; i<trainDescriptors.size(); i++){
        // ROS_INFO("%d\n", trainDescriptors[i].rows);
    // }
    // for(int i=0; i< matches.size(); i++)
    // {
        // ROS_INFO("%lf %d %d %d\n", matches[i].distance, matches[i].imgIdx, matches[i].queryIdx, matches[i].trainIdx);
    // }
}

static void markHomographies( const Mat& markedImage, const vector<Mat>& trainImages, const vector<KeyPoint>& queryKeypoints, 
                        const vector<vector<KeyPoint>>& trainKeypoints, const vector<DMatch>& matches )
{
    Scalar color(0, 255, 0); vector<Point2f> objCorners(4), sceneCorners(4), objPoints, scenePoints;
    for(int i=0; i<trainImages.size(); i++)
    {
        objPoints.clear(); scenePoints.clear();
        for(int j=0; j<matches.size(); j++)
        {
            if(matches[j].imgIdx != i) continue;
            objPoints.push_back(trainKeypoints[i][matches[j].trainIdx].pt);
            scenePoints.push_back(queryKeypoints[matches[j].queryIdx].pt);
        }
        // ROS_INFO("%d %d\n", objPoints.size(), scenePoints.size());
        if(objPoints.size() < 4 || scenePoints.size() < 4) continue;
        Mat H = findHomography(objPoints, scenePoints, CV_RANSAC);
        
        objCorners[0] = Point(0, 0); 
        objCorners[1] = Point(trainImages[i].cols, 0);
        objCorners[2] = Point(trainImages[i].cols, trainImages[i].rows);
        objCorners[3] = Point(0, trainImages[i].rows);
        perspectiveTransform(objCorners, sceneCorners, H);

        line(markedImage, sceneCorners[0], sceneCorners[1], color, 4);
        line(markedImage, sceneCorners[1], sceneCorners[2], color, 4); 
        line(markedImage, sceneCorners[2], sceneCorners[3], color, 4);
        line(markedImage, sceneCorners[3], sceneCorners[0], color, 4);
    }
}

static void saveResultImages( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,
                       const vector<Mat>& trainImages, const vector<vector<KeyPoint> >& trainKeypoints,
                       const vector<DMatch>& matches, const vector<string>& trainImagesNames, const string& resultDir )
{
    ROS_INFO("Save results...\n");
    Mat drawImg;
    vector<char> mask;
    for( size_t i = 0; i < trainImages.size(); i++ )
    {
        if( !trainImages[i].empty() )
        {
            maskMatchesByTrainImgIdx( matches, (int)i, mask );
            drawMatches( queryImage, queryKeypoints, trainImages[i], trainKeypoints[i],
                         matches, drawImg, Scalar(255, 0, 0), Scalar(0, 255, 255), mask );
            string filename = resultDir + "/res_" + trainImagesNames[i];
            if( !imwrite( filename, drawImg ) )
              ROS_ERROR("Image %s can not be saved (may be because directory %s does not exist).\n", filename, resultDir);
        }
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sifter");
    ros::NodeHandle nh;
    ros::Rate loopRate(10);

    nh.getParam("images", numTrainImages);
    nh.getParam("debug", debug);
    nh.getParam("stream", stream);

    ros::Subscriber imgSub = nh.subscribe("image_raw", 1, imageCb);
    ros::Publisher imgPub = nh.advertise<sensor_msgs::Image>("marked_image", 1);

    string detectorType = defaultDetectorType;
    string matcherType = defaultMatcherType;
    string fileWithTrainImages = defaultFileWithTrainImages;
    // string descriptorType = defaultDescriptorType;
    string queryImageName = defaultQueryImageName;
    string dirToSaveResImages = defaultDirToSaveResImages;

    // Ptr<FeatureDetector> featureDetector;
    // Ptr<DescriptorExtractor> descriptorExtractor;
    Ptr<Feature2D> detector; Ptr<DescriptorMatcher> descriptorMatcher;
    if( !createDetectorDescriptorMatcher( matcherType, detector, descriptorMatcher ) )  return -1;

    vector<Mat> trainImages; vector<string> trainImagesNames;
    if( !readTrainImages( fileWithTrainImages, trainImages, trainImagesNames ) ) return -1;

    vector<vector<KeyPoint> > trainKeypoints; vector<Mat> trainDescriptors;
    detectAndComputeTrainKeypoints( trainImages, trainKeypoints, trainDescriptors, detector );

    vector<KeyPoint> queryKeypoints; Mat queryDescriptors; vector<DMatch> matches;
    if(!stream) 
    {
        queryImage = imread( queryImageName, CV_LOAD_IMAGE_GRAYSCALE);
        markedImage = imread( queryImageName ); 
        detectAndComputeQueryKeypoints( queryImage, queryKeypoints, queryDescriptors, detector );
        matchDescriptors( queryDescriptors, trainDescriptors, matches, descriptorMatcher );
        markHomographies( markedImage, trainImages, queryKeypoints, trainKeypoints, matches);
        imwrite("out/marked.jpg", markedImage);
        saveResultImages( queryImage, queryKeypoints, trainImages, trainKeypoints,
                      matches, trainImagesNames, dirToSaveResImages );
    }
    else
    {
        while(ros::ok())
        {
            while(sceneID == 0) ros::spinOnce();
            detectAndComputeQueryKeypoints( queryImage, queryKeypoints, queryDescriptors, detector );
            matchDescriptors( queryDescriptors, trainDescriptors, matches, descriptorMatcher );
            markHomographies( markedImage, trainImages, queryKeypoints, trainKeypoints, matches);
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", markedImage).toImageMsg();
            imgPub.publish(msg);
            ros::spinOnce();
            loopRate.sleep();
        }
    }
    return 0;
}