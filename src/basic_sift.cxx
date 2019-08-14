#include <ros/ros.h>
#include <opencv-3.3.1-dev/opencv2/imgproc.hpp>
#include <opencv-3.3.1-dev/opencv2/calib3d/calib3d.hpp>
#include <opencv-3.3.1-dev/opencv2/features2d/features2d.hpp>
#include <opencv-3.3.1-dev/opencv2/core/core.hpp>
#include <opencv-3.3.1-dev/opencv2/highgui/highgui.hpp>
#include <opencv-3.3.1-dev/opencv2/xfeatures2d.hpp>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "feature_detector");
    cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();

    cv::Mat kout_1, kout_2, match, good_match, img_match;
    cv::Mat d_1, d_2;
    const cv::Mat img_1 = cv::imread("image1.jpg", 0);
    const cv::Mat img_2 = cv::imread("image2.jpg", 0);

    std::vector<cv::KeyPoint> kp_1, kp_2;
    f2d->detectAndCompute(img_1, cv::noArray(), kp_1, d_1);
    f2d->detectAndCompute(img_2, cv::noArray(), kp_2, d_2);
    
    cv::drawKeypoints(img_1, kp_1, kout_1);
    cv::imwrite("kp_1.jpg", kout_1);
    cv::drawKeypoints(img_2, kp_2, kout_2);
    cv::imwrite("kp_2.jpg", kout_2);

    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(d_1, d_2, matches);

    cv::drawMatches(img_1, kp_1, img_2, kp_2, matches, match);
    cv::imwrite("matches.jpg", match);

    double max_dist = 0, min_dist = 100;
    for(int i=0; i < d_1.rows; i++){
        double dist = matches[i].distance;
        if(dist<min_dist) min_dist = dist;
        if(dist>max_dist) max_dist = dist;
    }

    std::vector<cv::DMatch> good_matches;
    for(int i=0; i < d_1.rows; i++){
        if(matches[i].distance < 3*min_dist){
            good_matches.push_back(matches[i]);
        }
    }

    cv::drawMatches(img_1, kp_1, img_2, kp_2, good_matches, good_match, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("good_match.jpg", good_match);

    std::vector<cv::Point2f> obj, scene;
    for(int i=0; i<good_matches.size(); i++){
        obj.push_back(kp_1[good_matches[i].queryIdx].pt);
        scene.push_back(kp_2[good_matches[i].trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(obj, scene, CV_RANSAC);
    std::vector<cv::Point2f> obj_corners(4), scene_corners(4);
    obj_corners[0] = cv::Point(0,0);
    obj_corners[1] = cv::Point(img_1.cols, 0);
    obj_corners[2] = cv::Point(img_1.cols, img_1.rows);
    obj_corners[3] = cv::Point(0, img_1.rows); 
    
    cv::perspectiveTransform(obj_corners, scene_corners, H);
    cv::Scalar color(0,255,0);
    line(img_match, scene_corners[0] + cv::Point2f(img_1.cols, 0), scene_corners[1] + cv::Point2f(img_1.cols, 0), color, 4);
    line(img_match, scene_corners[1] + cv::Point2f(img_1.cols, 0), scene_corners[2] + cv::Point2f(img_1.cols, 0), color, 4); 
    line(img_match, scene_corners[2] + cv::Point2f(img_1.cols, 0), scene_corners[3] + cv::Point2f(img_1.cols, 0), color, 4);
    line(img_match, scene_corners[3] + cv::Point2f(img_1.cols, 0), scene_corners[0] + cv::Point2f(img_1.cols, 0), color, 4);
    
    cv::imwrite("image.jpg", img_match);
    return 0;
}
