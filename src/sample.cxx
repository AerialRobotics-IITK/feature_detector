#include <ros/ros.h>
#include <opencv-3.3.1-dev/opencv2/imgproc.hpp>
#include <opencv-3.3.1-dev/opencv2/calib3d/calib3d.hpp>
#include <opencv-3.3.1-dev/opencv2/features2d/features2d.hpp>
#include <opencv-3.3.1-dev/opencv2/core/core.hpp>
#include <opencv-3.3.1-dev/opencv2/highgui/highgui.hpp>
#include <opencv-3.3.1-dev/opencv2/xfeatures2d.hpp>

#define echo(x) std::cout << (x) << std::endl

typedef std::vector<cv::Mat> Mats;
typedef std::vector<cv::KeyPoint> points;
typedef std::vector<cv::DMatch> match;
typedef std::vector<match> nn_match;

cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();

int main(int argc, char** argv){
    ros::init(argc, argv, "feature_detector");
    ros::NodeHandle nh; 

    bool debug = false; nh.getParam("debug", debug);
    int n = 0; nh.getParam("images", n);

    Mats images; cv::Mat image;
    
    for(int i=1; i<=n; i++){
        image = cv::imread("/home/ashwin/imav_ws/src/feature_detector/in/object.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        images.push_back(image);
    }

    // reduce processing by precomputing and storing
    std::vector<points> keypoints; points kpts, obj_kpts;
    Mats descriptors; cv::Mat desc, obj_descs;
    
    for(int i=0; i<n; i++){
        detector->detectAndCompute(images.at(i), cv::noArray(), kpts, desc);
        keypoints.push_back(kpts); descriptors.push_back(desc);
    }
    cv::hconcat(desc, obj_descs);
    for(int i=0; i<keypoints.size(); i++){
        for(int j=0; j<keypoints[i].size(); j++){
            obj_kpts.push_back(keypoints[i].at(j));
        }
    }

    if(debug){
        cv::Mat output;
        for(int i=0; i<n; i++){
            cv::drawKeypoints(images.at(i), keypoints.at(i), output);
            cv::imshow("/home/ashwin/imav_ws/src/feature_detector/out/kpts_" + std::to_string(i+1), output);
            while(cv::waitKey(5) != 'q');
        }
    }
    std::cout << "finished loading database" << std::endl;

    // replace with loop here for a video scene
    cv::Mat scene = cv::imread("/home/ashwin/imav_ws/src/feature_detector/in/scene.jpg", CV_LOAD_IMAGE_GRAYSCALE); 
    cv::Mat scene_descs; points scene_kpts;
    detector->detectAndCompute(scene, cv::noArray(), scene_kpts, scene_descs);

    if(debug){
        cv::Mat output;
        cv::drawKeypoints(scene, scene_kpts, output);
        cv::imshow("/home/ashwin/imav_ws/src/feature_detector/out/scene_kpts", output);
        while(cv::waitKey(5) != 'q');
    }
    
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::KDTreeIndexParams>(5));
    nn_match matches;
    matcher.knnMatch(obj_descs, scene_descs, matches, 2);
    // how will you display matches from multiple images?

    match good_matches;
    for(int i=0; i<matches.size(); i++){
        if(matches[i][0].distance < 0.7*matches[i][1].distance){
            good_matches.push_back(matches[i][0]);
        }
    }

    std::vector<cv::Point2f> obj, scene_pts;
    for(int i=0; i<good_matches.size(); i++){
        printf("%lf %d %d %d\n", good_matches[i].distance, good_matches[i].imgIdx, good_matches[i].trainIdx, good_matches[i].queryIdx);
        obj.push_back(obj_kpts[good_matches[i].queryIdx].pt);
        scene_pts.push_back(scene_kpts[good_matches[i].trainIdx].pt);
    }

    // cv::Mat H = cv::findHomography(obj, scene_pts, CV_RANSAC);
    // std::vector<cv::Point2f> obj_corners(4), scene_corners(4);
    // obj_corners[0] = cv::Point(0,0);
    // obj_corners[1] = cv::Point(scene.cols, 0);
    // obj_corners[2] = cv::Point(scene.cols, scene.rows);
    // obj_corners[3] = cv::Point(0, scene.rows); 
    // cv::perspectiveTransform(obj_corners, scene_corners, H);
// 
    // cv::Scalar color(0,255,0);
    // line(scene, scene_corners[0], scene_corners[1], color, 4);
    // line(scene, scene_corners[1], scene_corners[2], color, 4); 
    // line(scene, scene_corners[2], scene_corners[3], color, 4);
    // line(scene, scene_corners[3], scene_corners[0], color, 4);
    
    cv::imwrite("out/image.jpg", scene);

}