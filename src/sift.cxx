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

cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();

int main(int argc, char** argv){
    ros::init(argc, argv, "feature_detector");
    ros::NodeHandle nh; 

    bool debug = false; nh.getParam("debug", debug);
    int n = 0; nh.getParam("images", n);

    Mats images; cv::Mat image;
    
    for(int i=1; i<=n; i++){
        image = cv::imread("/home/tanay/catkin_ws/src/feature_detector/in/image" + std::to_string(i) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
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
            cv::imshow("/home/tanay/catkin_ws/src/feature_detector/out/kpts_" + std::to_string(i+1), output);
            cv::waitKey(0);
        }
    }
    std::cout << "finished loading database" << std::endl;
    // replace with loop here for a video scene
    cv::Mat scene = cv::imread("/home/tanay/catkin_ws/src/feature_detector/in/scene.jpg", CV_LOAD_IMAGE_GRAYSCALE); 
    cv::Mat scene_desc; points scene_kpts;
    detector->detectAndCompute(scene, cv::noArray(), scene_kpts, scene_desc);

    if(debug){
        cv::Mat output;
        cv::drawKeypoints(scene, scene_kpts, output);
        cv::imshow("/home/tanay/catkin_ws/src/feature_detector/out/scene_kpts", output);
        cv::waitKey(0);
    }
    
    std::vector<double> delta_dist_temp;
    std::vector<std::vector<double> > delta_dist;
    std::vector<std::vector<std::vector<double> > > dist_mat;
    double temp_sum_sq = 0;

    for(int i=0;i<scene_desc.rows;i++)
    {
        delta_dist.clear();
        delta_dist.shrink_to_fit();
        for(int j=0;j<n;j++)
        {
            delta_dist_temp.clear();
            delta_dist_temp.shrink_to_fit();
            for(int k=0;k<descriptors.at(j).rows;k++)
            {
                temp_sum_sq = 0;
                for(int l=0;l<128;l++)
                {
                    temp_sum_sq += pow(scene_desc.at<float>(i, l) - descriptors.at(j).at<float>(k, l),2);
                }
                if(temp_sum_sq < 10E3)
                {
                    echo(i);
                    echo(j);
                    echo(k);
                }
                delta_dist_temp.push_back(sqrt(temp_sum_sq));
            }
            delta_dist.push_back(delta_dist_temp);
        }
        dist_mat.push_back(delta_dist);
    }

    std::vector<cv::KeyPoint> key(7);
    key[0] = scene_kpts.at(9);
    key[1] = scene_kpts.at(35);
    key[2] = scene_kpts.at(37);
    key[3] = scene_kpts.at(94);
    key[4] = scene_kpts.at(52);
    key[5] = scene_kpts.at(53);
    key[6] = scene_kpts.at(92);

    cv::Mat output;
    cv::drawKeypoints(scene, key, output);
    cv::imshow("BRUHHH", output);
    cv::waitKey(0);
    // echo(dist_mat.size());
    // for(int i=0;i<n;i++)
    // {
    //     for(int j=0;j<descriptors.at(i).rows;j++)
    //     {
    //         temp_sum_sq = 0;
    //         for(int k=0;k<descriptors.at(i).cols;k++)
    //         {
    //             temp_sum_sq += (scene_desc.at<float>(j, k) - descriptors.at(i).at<float>(j, k))*(scene_desc.at<float>(j, k) - descriptors.at(i).at<float>(j, k));
    //         }
    //         std::cout << temp_sum_sq << std::endl;
    //         delta_dist.at<float>(i, j) = sqrt(temp_sum_sq);
    //         std::cout << "1 iteration" << std::endl;
    //     }
    // }
}