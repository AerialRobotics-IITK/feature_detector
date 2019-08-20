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

    cv::Mat kout_1, kout_2, match1, match2, img_match, good_match;
    cv::Mat d_1, d_2;
    const cv::Mat img_1 = cv::imread("in/object.jpg", 0);
    const cv::Mat img_2 = cv::imread("in/scene.jpg", 0);

    std::vector<cv::KeyPoint> kp_1, kp_2;
    f2d->detectAndCompute(img_1, cv::noArray(), kp_1, d_1);
    f2d->detectAndCompute(img_2, cv::noArray(), kp_2, d_2);
    
    cv::drawKeypoints(img_1, kp_1, kout_1);
    cv::imwrite("out/kp_1.jpg", kout_1);
    cv::drawKeypoints(img_2, kp_2, kout_2);
    cv::imwrite("out/kp_2.jpg", kout_2);

    cv::BFMatcher matcher;
    std::vector<std::vector<cv::DMatch>> matches;
    matcher.knnMatch(d_1, d_2, matches, 2);

    std::vector<cv::DMatch> matches1, matches2;
    for(int i=0; i<matches.size(); i++){
        matches1.push_back(matches[i][0]);
        matches2.push_back(matches[i][1]);
    }

    cv::drawMatches(img_1, kp_1, img_2, kp_2, matches1, match1);
    cv::imwrite("out/matches1.jpg", match1);

    cv::drawMatches(img_1, kp_1, img_2, kp_2, matches2, match2);
    cv::imwrite("out/matches2.jpg", match2);

    cv::drawMatches(img_1, kp_1, img_2, kp_2, matches, img_match, cv::Scalar(0,255,0), cv::Scalar(255,0,0), std::vector<std::vector<char>>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("out/matches.jpg", img_match);

    // make ratio a parameter
    std::vector<std::vector<cv::DMatch>> good_matches;
    for(int i=0; i < matches.size(); i++){
        printf("%lf %lf\n", matches[i][0].distance, matches[i][1].distance);
        if(matches[i][0].distance < 0.3*matches[i][1].distance)
            good_matches.push_back(matches[i]);
    }

    std::vector<cv::DMatch> good_matches1;
    for(int i=0; i < good_matches.size(); i++){
        good_matches1.push_back(good_matches[i][0]);
    }

    cv::drawMatches(img_1, kp_1, img_2, kp_2, good_matches1, good_match);
    cv::imwrite("out/good_match.jpg", good_match);

    std::vector<cv::Point2f> obj, scene;
    for(int i=0; i<good_matches.size(); i++){
        obj.push_back(kp_1[good_matches1[i].queryIdx].pt);
        scene.push_back(kp_2[good_matches1[i].trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(obj, scene, CV_RANSAC);
    for(int i=0; i<H.rows; i++){
        for(int j=0; j<H.cols; j++){
            printf("%lf ", H.at<double>(i,j));
        }
        printf("\n");
    }
    std::vector<cv::Point2f> obj_corners(4), scene_corners(4);
    obj_corners[0] = cv::Point(0,0);
    obj_corners[1] = cv::Point(img_1.cols, 0);
    obj_corners[2] = cv::Point(img_1.cols, img_1.rows);
    obj_corners[3] = cv::Point(0, img_1.rows); 
    
    cv::perspectiveTransform(obj_corners, scene_corners, H);
    for(int i=0; i<4; i++){
        printf("%lf %lf\n", scene_corners.at(i).x, scene_corners.at(i).y);
    }
    cv::Scalar color(0,255,0);
    line(img_2, scene_corners[0], scene_corners[1], color, 4);
    line(img_2, scene_corners[1], scene_corners[2], color, 4); 
    line(img_2, scene_corners[2], scene_corners[3], color, 4);
    line(img_2, scene_corners[3], scene_corners[0], color, 4);
    
    cv::imwrite("out/image.jpg", img_2);
    return 0;
}