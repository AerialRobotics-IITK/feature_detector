#include <ros/ros.h>
#include <opencv-3.3.1-dev/opencv2/imgproc.hpp>
#include <opencv-3.3.1-dev/opencv2/calib3d/calib3d.hpp>
#include <opencv-3.3.1-dev/opencv2/features2d/features2d.hpp>
#include <opencv-3.3.1-dev/opencv2/core/core.hpp>
#include <opencv-3.3.1-dev/opencv2/highgui/highgui.hpp>
#include <opencv-3.3.1-dev/opencv2/xfeatures2d.hpp>

typedef std::vector<cv::Mat> Mats;
typedef std::vector<cv::KeyPoint> points;
typedef std::vector<cv::DMatch> match;

cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();

int main(int argc, char** argv){
    ros::init(argc, argv, "feature_detector");
    ros::NodeHandle nh; 

    bool debug = false; nh.getParam("debug", debug);
    int n = 0; nh.getParam("images", n);
    Mats images; cv::Mat image;
    for(int i=1; i<=n; i++){
        image = cv::imread("in/image" + std::to_string(i) + ".jpg", 0);
        images.push_back(image);
    }

    std::vector<points> keypoints; points kpts;
    Mats descriptors; cv::Mat desc;
    for(int i=0; i<n; i++){
        detector->detectAndCompute(images.at(i), cv::noArray(), kpts, desc);
        keypoints.push_back(kpts); descriptors.push_back(desc);
    }

    if(debug){
        cv::Mat output;
        for(int i=0; i<n; i++){
            cv::drawKeypoints(images.at(i), keypoints.at(i), output);
            cv::imwrite("out/kpts_" + std::to_string(i+1) + ".jpg", output);
        }
    }

    // replace with loop here for a video scene
    cv::Mat scene = cv::imread("in/scene.jpg", 0); 
    cv::Mat scene_desc; points scene_kpts;
    detector->detectAndCompute(scene, cv::noArray(), scene_kpts, scene_desc);
    if(debug){
        cv::Mat output;
        cv::drawKeypoints(scene, scene_kpts, output);
        cv::imwrite("out/scene_kpts.jpg", output);
    }   
}