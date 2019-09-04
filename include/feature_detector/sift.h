#include <ros/ros.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>
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
#include <iostream>
#include <fstream>
#define check(X) std::cout << X << std::endl

struct imgData
{
    int cols;
    int rows;
    int idx;
    std::vector<int> matchIdxs;
    imgData(int cols_, int rows_, int idx_, std::vector<int> matchIdxs_) : cols(cols_), rows(rows_), idx(idx_), matchIdxs(matchIdxs_){}
};

const std::string defaultDetectorType = "SIFT";
const std::string defaultMatcherType = "FlannBased";
const std::string defaultQueryImageName = "in/scene.jpg";
const std::string defaultFileWithTrainImages = "in/train.txt";
const std::string defaultDirToSaveResImages = "out";
// const std::string defaultDescriptorType = "SIFT";

int sceneID = 0, numTrainImages = 0, queryPoints = 0, numObjects = 0;
std::vector<int> numMatches, imageIDs, objectIDs;
bool debug = true, stream = true;
double ratio = 0.7;
int min_points = 10;
std::vector<std::array<cv::Point2f, 4>> objects;
std::vector<cv::Point> centres;
cv::Mat queryImage, markedImage, keyedImage, greyedImage;
std::vector<imgData> bestImages;

static void maskMatchesByTrainImgIdx( const std::vector<cv::DMatch>& matches, int trainImgIdx, std::vector<char>& mask )
{
    mask.resize( matches.size() );
    std::fill( mask.begin(), mask.end(), 0 );
    for( size_t i = 0; i < matches.size(); i++ )
    {
        if( matches[i].imgIdx == trainImgIdx )
            mask[i] = 1;
    }
}

static void readTrainFilenames( const std::string& filename, std::string& dirName, std::vector<std::string>& trainFilenames )
{
    trainFilenames.clear();

    std::ifstream file( filename.c_str() );
    if ( !file.is_open() ){
        // printf("%s\n", filename.c_str());
        return;
    }
    std::size_t pos = filename.rfind('\\');
    char dlmtr = '\\';
    if (pos == std::string::npos)
    {
        pos = filename.rfind('/');
        dlmtr = '/';
    }
    dirName = pos == std::string::npos ? "" : filename.substr(0, pos) + dlmtr;

    while( !file.eof() )
    {
        std::string str; getline( file, str );
        if( str.empty() ) break;
        trainFilenames.push_back(str);
    }
    file.close();
}

static bool readTrainImages( const std::string& trainFilename, std::vector<cv::Mat>& trainImages, std::vector<std::string>& trainImageNames )
{
    // ROS_INFO("< Reading the images...\n");
    // queryImage = imread( queryImageName, CV_LOAD_IMAGE_GRAYSCALE);
    // if( queryImage.empty() )
    // {
        // ROS_ERROR("Query image can not be read. \n");
        // return false;
    // }

    std::string trainDirName;
    readTrainFilenames( trainFilename, trainDirName, trainImageNames );
    if( trainImageNames.empty() )
    {
        ROS_ERROR("Train image filenames can not be read.>\n");
        return false;
    }
    int readImageCount = 0;
    for( size_t i = 0; i < trainImageNames.size(); i++ )
    {
        std::string filename = trainDirName + trainImageNames[i];
        cv::Mat img = cv::imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
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

static void detectAndComputeTrainKeypoints( const std::vector<cv::Mat>& trainImages, std::vector<std::vector<cv::KeyPoint> >& trainKeypoints, 
                                    std::vector<cv::Mat>& trainDescriptors, cv::Ptr<cv::Feature2D>& detector )
{
    //   ROS_INFO("< Computing descriptors for training images...\n");
    detector->detect( trainImages, trainKeypoints );
    detector->compute( trainImages, trainKeypoints, trainDescriptors); 
    int totalTrainDesc = 0;
    if(debug)
    {
        for( std::vector<cv::Mat>::const_iterator tdIter = trainDescriptors.begin(); tdIter != trainDescriptors.end(); tdIter++ )
            totalTrainDesc += tdIter->rows;
        ROS_INFO("Total train descriptors count: %d", totalTrainDesc );
        cv::Mat output;
        for(int i = 0; i < trainImages.size(); i++)
        {
            drawKeypoints(trainImages[i], trainKeypoints[i], output);
            imwrite("out/kpts_" + std::to_string(i) + ".jpg", output);
        }
    }
}

static void detectAndComputeQueryKeypoints( const cv::Mat& queryImage, std::vector<cv::KeyPoint>& queryKeypoints,
                                    cv::Mat& queryDescriptors, cv::Ptr<cv::Feature2D> detector)
{
    //   ROS_INFO("< Computing descriptors for query image...\n");
    // queryKeypoints.clear();
    detector->detectAndCompute( queryImage, cv::noArray(), queryKeypoints, queryDescriptors );
    if(debug)
    {
        queryPoints = queryDescriptors.rows;
    }
}

static void matchDescriptors( const cv::Mat& queryDescriptors, const std::vector<cv::Mat>& trainDescriptors,
                       std::vector<cv::DMatch>& matches, cv::FlannBasedMatcher descriptorMatcher )
{
    // ROS_INFO("< Set train descriptors collection in the matcher and match query descriptors to them...\n");
    cv::TickMeter tm;
    std::vector<std::vector<cv::DMatch>> nn_matches;
    matches.clear();

    tm.start();
    descriptorMatcher.add( trainDescriptors );
    descriptorMatcher.train();
    tm.stop();
    double buildTime = tm.getTimeMilli();

    tm.start();
    descriptorMatcher.knnMatch( queryDescriptors, nn_matches, 2);
    tm.stop();
    double matchTime = tm.getTimeMilli();

    for(int i=0; i<nn_matches.size(); i++){
        if(nn_matches[i][0].distance < ratio*nn_matches[i][1].distance){
            matches.push_back(nn_matches[i][0]);
        }
    }

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

static void saveResultImages( const cv::Mat& queryImage, const std::vector<cv::KeyPoint>& queryKeypoints,
                       const std::vector<cv::Mat>& trainImages, const std::vector<std::vector<cv::KeyPoint> >& trainKeypoints,
                       const std::vector<cv::DMatch>& matches, const std::vector<std::string>& trainImagesNames, const std::string& resultDir )
{
    ROS_INFO("Save results...\n");
    cv::Mat drawImg;
    std::vector<char> mask;
    for( size_t i = 0; i < trainImages.size(); i++ )
    {
        if( !trainImages[i].empty() )
        {
            maskMatchesByTrainImgIdx( matches, (int)i, mask );
            drawMatches( queryImage, queryKeypoints, trainImages[i], trainKeypoints[i],
                         matches, drawImg, cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 255), mask );
            std::string filename = resultDir + "/res_" + trainImagesNames[i];
            if( !imwrite( filename, drawImg ) )
              ROS_ERROR("Image %s can not be saved (may be because directory %s does not exist).\n", filename, resultDir);
        }
    }
}

// static void detectKeypoints( const cv::Mat& queryImage, std::vector<cv::KeyPoint>& queryKeypoints,
//                       const std::vector<cv::Mat>& trainImages, std::vector<std::vector<cv::KeyPoint> >& trainKeypoints,
//                       Ptr<FeatureDetector>& featureDetector )
// {
    // ROS_INFO("< Extracting keypoints from images...\n");
//     featureDetector->detect( queryImage, queryKeypoints );
    // featureDetector->detect( trainImages, trainKeypoints );
// }

// static void computeDescriptors( const cv::Mat& queryImage, std::vector<cv::KeyPoint>& queryKeypoints, cv::Mat& queryDescriptors,
//                          const std::vector<cv::Mat>& trainImages, std::vector<std::vector<cv::KeyPoint> >& trainKeypoints, std::vector<cv::Mat>& trainDescriptors,
//                          Ptr<DescriptorExtractor>& descriptorExtractor )
// {
//     ROS_INFO("< Computing descriptors for keypoints...\n");
//     descriptorExtractor->compute( queryImage, queryKeypoints, queryDescriptors );
//     descriptorExtractor->compute( trainImages, trainKeypoints, trainDescriptors );

//     int totalTrainDesc = 0;
//     for( std::vector<cv::Mat>::const_iterator tdIter = trainDescriptors.begin(); tdIter != trainDescriptors.end(); tdIter++ )
//         totalTrainDesc += tdIter->rows;

//      ROS_INFO("Query descriptors count: %d; Total train descriptors count: %d", queryDescriptors.rows, totalTrainDesc );
// }

// static bool createDetectorDescriptorMatcher( 
//                                       const std::string& matcherType,
//                                       Ptr<Feature2D>& detector,
//                                       Ptr<DescriptorMatcher>& descriptorMatcher )
// {
//     // ROS_INFO("< Creating feature detector, descriptor extractor and descriptor matcher ...\n");
//     // featureDetector = FeatureDetector::create( detectorType );
//     // descriptorExtractor = DescriptorExtractor::create( descriptorType );
//     detector = xfeatures2d::SIFT::create();
//     descriptorMatcher = DescriptorMatcher::create( matcherType );

//     bool isCreated = !( detector.empty() || descriptorMatcher.empty() );
//     if( !isCreated )    ROS_ERROR("Can not create feature detector or descriptor extractor or descriptor matcher of given types.\n");

//     return isCreated;
// }


// static void printPrompt( const std::string& applName )
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