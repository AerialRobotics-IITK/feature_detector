#include <feature_detector/sift.h>

static void imageCb(const sensor_msgs::Image msg)
{
    if(run){ 
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

        sceneID = msg.header.seq;
        if(debug)
        {
          markedImage = cv_ptr->image;
          cv::cvtColor(markedImage, queryImage, cv::COLOR_BGR2GRAY);
          greyedImage = queryImage;
        }
    }
}

static void odomCb(const nav_msgs::Odometry msg){if(run) odom = msg;}

static void findBestImages( const std::vector<cv::Mat>& trainImages, const std::vector<cv::DMatch>& matches )
{
    numMatches.clear();
    bestImages.clear(); int matchNum[numTrainImages]; int imgNum = 0;
    for(int i = 0; i < numTrainImages; i++) matchNum[i] = 0;
    for(int i = 0; i < matches.size(); i++)
    {
        imgNum = matches[i].imgIdx;
        matchNum[imgNum] = matchNum[imgNum] + 1;
    }
    // printf("%d %d match\n", matchNum[0], matchNum[1]);
    for(int i = 0; i < numTrainImages; i++) numMatches.push_back(matchNum[i]);
    int objID = imageIDs[0], idx = 0, i = 0, maxMatches = 0;
    std::vector<int> imgMatches;
    while(i < numTrainImages)
    {
        // add && imageIDs[i] == objID for best image of each type
        while(i < numTrainImages)
        {
            if(maxMatches < matchNum[i])
            {
                maxMatches = matchNum[i];
                idx = i;
            }
            objID = imageIDs[i]; i++;
        }
        // printf("%d %d find\n", idx, maxMatches);
        imgMatches.clear();
        for(int j = 0; j < matches.size(); j++) if(matches[j].imgIdx == idx) imgMatches.push_back(j);
        // printf("%d bef\n", bestImages.size());
        bestImages.push_back(imgData(trainImages[idx].cols, trainImages[idx].rows, idx, imgMatches));
        // printf("%d aft\n", bestImages.size());
        objID = imageIDs[i]; maxMatches = 0;
    }
    return;
}

bool sizeCheck( const std::array<cv::Point2f, 4> corners, int id)
{
    areas.clear(); bool lengthPassed = true;
    double lx = 0, ly = 0;

    lx = fabs(corners.at(1).x - corners.at(0).x) + fabs(corners.at(2).x - corners.at(3).x);
    ly = fabs(corners.at(3).y - corners.at(0).y) + fabs(corners.at(2).y - corners.at(1).y);

    lengthPassed = (fabs(bestImages[id].rows - lx/2) <= lengthError) && (fabs(bestImages[id].cols - ly/2) <= lengthError);

    if(!sizeCheckFlag) return lengthPassed;
    else
    {
        double area = 0;
        bool sizePassed = true;
        for(int i=0; i<4; i++)
        {
           area += corners.at(i%4).x * corners.at((i+1)%4).y;
           area -= corners.at(i%4).y * corners.at((i+1)%4).x;
        }
        area = fabs(area/2);
        sizePassed = (fabs(area - pixelSizes[bestImages[id].idx]) <= sizeError);
        if(sizePassed && lengthPassed) areas.push_back(area);
        return sizePassed && lengthPassed;
    }
}

static void findHomographies( const std::vector<cv::KeyPoint>& queryKeypoints, const std::vector<std::vector<cv::KeyPoint>>& trainKeypoints, 
                                                                        const std::vector<cv::DMatch>& matches )
{
    std::vector<cv::Point2f> objCorners(4), sceneCorners(4), objPoints, scenePoints;
    std::vector<cv::KeyPoint> matchPoints;
    objects.clear(); objectIDs.clear();
    for(int i=0; i < bestImages.size(); i++)
    {
        objPoints.clear(); scenePoints.clear();
        for(int j=0; j < bestImages[i].matchIdxs.size(); j++)
        {
            // if(matches[bestImages[i].matchIdxs[j]].imgIdx != bestImages[i].idx) ROS_ERROR("SoMeThinG wEnt WrOng!");
            objPoints.push_back(trainKeypoints[bestImages[i].idx][matches[bestImages[i].matchIdxs[j]].trainIdx].pt);
            scenePoints.push_back(queryKeypoints[matches[bestImages[i].matchIdxs[j]].queryIdx].pt);
            if(debug) matchPoints.push_back(queryKeypoints[matches[bestImages[i].matchIdxs[j]].queryIdx]);
        }
        // ROS_INFO("%d %d\n", objPoints.size(), scenePoints.size());
        // numMatches.push_back(objPoints.size());
        // if(objPoints.size() != numMatches[bestImages[i].idx] || scenePoints.size() != numMatches[bestImages[i].idx]) ROS_ERROR("NOpe");
        if(objPoints.size() < min_points || scenePoints.size() < min_points) continue;
        cv::Mat H = cv::findHomography(objPoints, scenePoints, CV_RANSAC, 4);
        
        if(H.empty()) continue;
        objCorners[0] = cv::Point(0, 0); 
        objCorners[1] = cv::Point(bestImages[i].cols, 0);
        objCorners[2] = cv::Point(bestImages[i].cols, bestImages[i].rows);
        objCorners[3] = cv::Point(0, bestImages[i].rows);
        cv::perspectiveTransform(objCorners, sceneCorners, H);

        std::array<cv::Point2f, 4> corners;
        for(int n = 0; n < 4; n++) corners[n] = sceneCorners[n];
        // printf("%lf %lf %lf %lf -1\n", sceneCorners[0].x, sceneCorners[1].x, sceneCorners[2].x, sceneCorners[3].x);
        // printf("%lf %lf %lf %lf 1\n", corners[0].y, corners[1].y, corners[2].y, corners[3].y);
        // printf("%d bef\n", objects.size());
        if(sizeCheck(corners, i))
        {
            objects.push_back(corners);
            // printf("%d aft\n", objects.size());
            objectIDs.push_back(bestImages[i].idx);
        }
    }

    if(debug) cv::drawKeypoints(queryImage, matchPoints, keyedImage);
    return;
}

static void findCentres()
{
    double cent_x = 0, cent_y = 0;
    centres.clear();
    for(int i=0; i<objects.size(); i++)
    {
        cent_x = cent_y = 0;
        for(int n = 0; n < 4; n++)
        {
            cent_x += objects[i][n].x;
            cent_y += objects[i][n].y;
        }
        cv::Point2f centre(cent_x/4.0, cent_y/4.0);
        centres.push_back(centre);
    }
}

static void markHomographies( const cv::Mat& markedImage )
{
    for(int i=0; i<objects.size(); i++)
    {
        cv::Scalar color((16*i)%255, 255, (32*i)%255);
        cv::line(markedImage, objects[i][0], objects[i][1], color, 4);
        cv::line(markedImage, objects[i][1], objects[i][2], color, 4); 
        cv::line(markedImage, objects[i][2], objects[i][3], color, 4);
        cv::line(markedImage, objects[i][3], objects[i][0], color, 4);
        cv::circle(markedImage, centres[i], 2, (255,0,0), -1);
        cv::putText(markedImage, std::to_string(objectIDs[i]), centres[i], cv::FONT_HERSHEY_SIMPLEX, 2, color, 4);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sifter");
    ros::NodeHandle sh, ph("~");
    ros::Rate loopRate(10);
    // double f_time;

    loadParams(ph);

    ros::ServiceServer exec_server = ph.advertiseService("terminate", serviceCall);

    ros::Subscriber imgSub = sh.subscribe<sensor_msgs::Image>("image_raw", 1, imageCb);
    ros::Subscriber odomSub = sh.subscribe<nav_msgs::Odometry>("odom", 10, odomCb);

    ros::Publisher imgPub = ph.advertise<sensor_msgs::Image>("marked_image", 1);
    ros::Publisher keyPub = ph.advertise<sensor_msgs::Image>("keyed_image", 1);
    ros::Publisher matchPub = ph.advertise<std_msgs::Int32MultiArray>("matches", 1);
    ros::Publisher countPub = ph.advertise<std_msgs::Int32>("keypoints", 1);
    ros::Publisher greyPub = ph.advertise<sensor_msgs::Image>("greyed_image", 1);
    ros::Publisher objPub = sh.advertise<mav_utils_msgs::BBPoses>("object_poses", 1);
    // ros::Publisher timePub = ph.advertise<std_msgs::Float64MultiArray>("times",1);

    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();
    // cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SURF::create();

    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::KDTreeIndexParams>(5));

    std::vector<cv::Mat> trainImages; std::vector<std::string> trainImagesNames;
    if( !readTrainImages( fileWithTrainImages, trainImages, trainImagesNames ) ) return -1;

    std::vector<std::vector<cv::KeyPoint>> trainKeypoints; std::vector<cv::Mat> trainDescriptors;
    detectAndComputeTrainKeypoints( trainImages, trainKeypoints, trainDescriptors, detector );

    std::vector<cv::KeyPoint> queryKeypoints; cv::Mat queryDescriptors; std::vector<cv::DMatch> matches;

    while(ros::ok() && !(exit))
    {
        if(run)
        {
          while(sceneID == 0 && odom.pose.pose.position.z == 0) ros::spinOnce();

            // times.clear();
            detectAndComputeQueryKeypoints( queryImage, queryKeypoints, queryDescriptors, detector );
            matchDescriptors( queryDescriptors, trainDescriptors, matches, matcher );
            findBestImages( trainImages, matches );
            findHomographies( queryKeypoints, trainKeypoints, matches );
            findCentres();
            if(debug) markHomographies( markedImage );

            mav_utils_msgs::BBPoses pose_msg = findPoses( &centres );
            pose_msg.imageID = sceneID; pose_msg.stamp = ros::Time::now();
            if(!pose_msg.object_poses.empty()) objPub.publish(pose_msg);

            if(debug)
            {
                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", markedImage).toImageMsg();
                sensor_msgs::ImagePtr key_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", keyedImage).toImageMsg();
                sensor_msgs::ImagePtr grey_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", greyedImage).toImageMsg();
                std_msgs::Int32MultiArray match_msg; std_msgs::Int32 keypts_msg;
                // std_msgs::Float64MultiArray time_msg;
                keypts_msg.data = queryPoints;
                // for(int i = 0; i < times.size(); i++) time_msg.data.push_back(times[i]);
                for (int i = 0; i < numMatches.size(); i++) match_msg.data.push_back(numMatches[i]);
                greyPub.publish(grey_msg); imgPub.publish(msg); keyPub.publish(key_msg); countPub.publish(keypts_msg); matchPub.publish(match_msg);
                // timePub.publish(time_msg);
            }
        }

        ros::spinOnce();
        loopRate.sleep();
    }

    return 0;
}