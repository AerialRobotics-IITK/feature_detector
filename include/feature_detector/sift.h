#include <ros/ros.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>
#include <mav_utils_msgs/BBPose.h>
#include <mav_utils_msgs/BBPoses.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_datatypes.h>

#include <opencv-3.3.1-dev/opencv2/highgui/highgui.hpp>
#include <opencv-3.3.1-dev/opencv2/features2d/features2d.hpp>
#include <opencv-3.3.1-dev/opencv2/core/core.hpp>
#include <opencv-3.3.1-dev/opencv2/imgproc/imgproc.hpp>
#include <opencv-3.3.1-dev/opencv2/objdetect/objdetect.hpp>
#include <opencv-3.3.1-dev/opencv2/xfeatures2d.hpp>
#include <opencv-3.3.1-dev/opencv2/opencv.hpp>
#include <opencv-3.3.1-dev/opencv2/opencv_modules.hpp>

#define check(X) std::cout << X << std::endl

struct imgData
{
    int cols;
    int rows;
    int idx;
    std::vector<int> matchIdxs;
    imgData(int cols_, int rows_, int idx_, std::vector<int> matchIdxs_) : cols(cols_), rows(rows_), idx(idx_), matchIdxs(matchIdxs_){}
};

std::string fileWithTrainImages = "in/train.txt";
std::string dirToSaveResImages = "out";

int sceneID = 0, numTrainImages = 0;
int queryPoints = 0, numObjects = 0;
int min_points = 10;

double sizeError = 100, lengthError = 20;
double ratio = 0.7;

bool debug = true, sizeCheckFlag = false;

std::vector<int> numMatches, imageIDs, objectIDs, pixelSizes;
std::vector<double> areas;
std::vector<cv::Point2f> centres;
std::vector<std::array<cv::Point2f, 4>> objects;
std::vector<imgData> bestImages;

cv::Mat intrinsic = cv::Mat_<double>(3,3);
cv::Mat distCoeffs = cv::Mat_<double>(1,5);
cv::Mat queryImage, markedImage, keyedImage, greyedImage;

Eigen::Matrix3f camMatrix, invCamMatrix, camToQuad, quadToCam;
Eigen::Vector3f tCam;

nav_msgs::Odometry odom;

static void loadParams(ros::NodeHandle nh)
{
    std::vector<double> tempList;
    int tempIdx=0;

    nh.getParam("ratio", ratio);    
    nh.getParam("images", numTrainImages);
    nh.getParam("debug", debug);
    nh.getParam("minPoints", min_points);
    nh.getParam("sizeCheck", sizeCheckFlag);
    nh.getParam("path/train", fileWithTrainImages);
    nh.getParam("path/save", dirToSaveResImages);
    nh.getParam("objects", numObjects);
    nh.getParam("error/size", sizeError);
    nh.getParam("error/length", lengthError);
    nh.getParam("objectIDs", imageIDs);
    nh.getParam("sizes", pixelSizes);

    nh.getParam("camera/translation", tempList);
    for (int i = 0; i < 3; i++)
    {
        tCam(i) = tempList[i];
    }

    nh.getParam("camera/rotation", tempList);
    tempIdx = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            quadToCam(i,j) = tempList[tempIdx++];
        }
    }

    nh.getParam("distortion_coefficients/data", tempList);
    for(int i=0; i<5; i++)
    {
        distCoeffs.at<double>(i) = tempList[i];
    }

    nh.getParam("camera_matrix/data", tempList);
    tempIdx=0;
    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            intrinsic.at<double>(i,j) = tempList[tempIdx++];
        }
    }
        for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            camMatrix(i,j) = intrinsic.at<double>(i,j);
        }
    }

    invCamMatrix = camMatrix.inverse();
    camToQuad = quadToCam.inverse();

    std::cout << "Parameters Loaded." << std::endl;

    return;
}

mav_utils_msgs::BBPoses findPoses(std::vector<cv::Point2f> *ptr)
{
    mav_utils_msgs::BBPoses msg;
    Eigen::Matrix3f scaleUp, quadToGlob;

    tf::Quaternion q1(odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w);
    Eigen::Quaternionf quat = Eigen::Quaternionf(q1.w(), q1.x(), q1.y(), q1.z());
    quadToGlob = quat.toRotationMatrix();

    for (int i=0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            if(i==j) scaleUp(i,j) = odom.pose.pose.position.z;
            else scaleUp(i,j) = 0;
        }
    }

    for(int i=0; i<ptr->size(); i++)
    {
        mav_utils_msgs::BBPose temp;

        temp.boxID = i;
        temp.store = true;
        temp.type = (imageIDs.at(i)) ? 42: 69;
        temp.area = (sizeCheckFlag) ? areas[i] : -1;

        Eigen::Vector3f imgVec(ptr->at(i).x,ptr->at(i).y,1);
        Eigen::Vector3f quadCoord = (camToQuad*scaleUp*invCamMatrix*imgVec) + tCam;
        Eigen::Vector3f globCoord = quadToGlob*quadCoord;

        temp.position.x = globCoord(0) + odom.pose.pose.position.x;
        temp.position.y = globCoord(1) + odom.pose.pose.position.y;
        temp.position.z = globCoord(2) + odom.pose.pose.position.z;

        msg.object_poses.push_back(temp);
    }

    return msg;
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
        trainImages.push_back(img);
    }
    if(!readImageCount)
    {
        ROS_ERROR("All train images could not be read.>\n");
        return false;
    }

    return true;
}

static void detectAndComputeTrainKeypoints( const std::vector<cv::Mat>& trainImages, std::vector<std::vector<cv::KeyPoint> >& trainKeypoints, 
                                    std::vector<cv::Mat>& trainDescriptors, cv::Ptr<cv::Feature2D>& detector )
{
    detector->detect( trainImages, trainKeypoints );
    detector->compute( trainImages, trainKeypoints, trainDescriptors ); 
    int totalTrainDesc = 0;
    if(debug)
    {
        for( std::vector<cv::Mat>::const_iterator tdIter = trainDescriptors.begin(); tdIter != trainDescriptors.end(); tdIter++ )
            totalTrainDesc += tdIter->rows;
        ROS_INFO( "Total train descriptors count: %d", totalTrainDesc );
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
    detector->detectAndCompute( queryImage, cv::noArray(), queryKeypoints, queryDescriptors );
    if(debug) queryPoints = queryDescriptors.rows;
}

static void matchDescriptors( const cv::Mat& queryDescriptors, const std::vector<cv::Mat>& trainDescriptors,
                       std::vector<cv::DMatch>& matches, cv::FlannBasedMatcher descriptorMatcher )
{
    std::vector<std::vector<cv::DMatch>> nn_matches;
    matches.clear();
    descriptorMatcher.add( trainDescriptors );
    descriptorMatcher.train();
    descriptorMatcher.knnMatch( queryDescriptors, nn_matches, 2 );
    for(int i=0; i<nn_matches.size(); i++)
    {
        if(nn_matches[i][0].distance < ratio*nn_matches[i][1].distance)
        {
            matches.push_back(nn_matches[i][0]);
        }
    }
}
