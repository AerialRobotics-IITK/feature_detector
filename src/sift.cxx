#include <feature_detector/sift.h>

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
    cv::cvtColor(markedImage, queryImage, cv::COLOR_BGR2GRAY);
    greyedImage = queryImage;
    sceneID = msg.header.seq;
    return;
}

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
        objects.push_back(corners);
        // printf("%d aft\n", objects.size());
        objectIDs.push_back(bestImages[i].idx);
    }

    if(debug) cv::drawKeypoints(queryImage, matchPoints, keyedImage);
    return;
}

static void markHomographies( const cv::Mat& markedImage )
{
    double cent_x = 0, cent_y = 0;
    centres.clear();
    for(int i=0; i<objects.size(); i++){
        cv::Scalar color((16*i)%255, 255, (32*i)%255);
        cent_x = cent_y = 0;
        for(int n = 0; n < 4; n++){
            cent_x += objects[i][n].x;
            cent_y += objects[i][n].y;
        }
        cv::Point centre(cent_x/4, cent_y/4);
        centres.push_back(centre);
        cv::line(markedImage, objects[i][0], objects[i][1], color, 4);
        cv::line(markedImage, objects[i][1], objects[i][2], color, 4); 
        cv::line(markedImage, objects[i][2], objects[i][3], color, 4);
        cv::line(markedImage, objects[i][3], objects[i][0], color, 4);
        cv::circle(markedImage, centre, 2, (255,0,0), -1);
        cv::putText(markedImage, std::to_string(objectIDs[i]), centre, cv::FONT_HERSHEY_SIMPLEX, 2, color, 4);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "sifter");
    ros::NodeHandle nh, ph("~");
    ros::Rate loopRate(10);

    // std::string detectorType = defaultDetectorType;
    // std::string matcherType = defaultMatcherType;
    // std::string descriptorType = defaultDescriptorType;
    std::string queryImageName = defaultQueryImageName;
    std::string fileWithTrainImages = defaultFileWithTrainImages;
    std::string dirToSaveResImages = defaultDirToSaveResImages;
    
    ph.getParam("ratio", ratio);
    ph.getParam("images", numTrainImages);
    // printf("%d", numTrainImages);
    ph.getParam("debug", debug);
    ph.getParam("minPoints", min_points);
    ph.getParam("stream", stream);
    ph.getParam("path/train", fileWithTrainImages);
    ph.getParam("path/save", dirToSaveResImages);
    ph.getParam("objects", numObjects);
    ph.getParam("objectIDs", imageIDs);

    ros::Subscriber imgSub = nh.subscribe("image_raw", 1, imageCb);
    ros::Publisher imgPub = nh.advertise<sensor_msgs::Image>("marked_image", 1);
    ros::Publisher keyPub = nh.advertise<sensor_msgs::Image>("keyed_image", 1);
    ros::Publisher matchPub = nh.advertise<std_msgs::Int32MultiArray>("matches", 1);
    ros::Publisher countPub = nh.advertise<std_msgs::Int32>("keypoints", 1);
    ros::Publisher greyPub = nh.advertise<sensor_msgs::Image>("greyed_image", 1);


    // cv::Ptr<cv::FeatureDetector> featureDetector;
    // cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::KDTreeIndexParams>(5));
    std::vector<std::array<cv::Point2f, 4> > objects; std::vector<cv::Point> centres;

    // if( !createDetectorDescriptorMatcher( matcherType, detector, descriptorMatcher ) )  return -1;

    std::vector<cv::Mat> trainImages; std::vector<std::string> trainImagesNames;
    if( !readTrainImages( fileWithTrainImages, trainImages, trainImagesNames ) ) return -1;

    // printf("%d in \n", trainImages.size());

    std::vector<std::vector<cv::KeyPoint>> trainKeypoints; std::vector<cv::Mat> trainDescriptors;
    detectAndComputeTrainKeypoints( trainImages, trainKeypoints, trainDescriptors, detector );

    std::vector<cv::KeyPoint> queryKeypoints; cv::Mat queryDescriptors; std::vector<cv::DMatch> matches;
    if(!stream) 
    {
        // queryImage = cv::imread( queryImageName, CV_LOAD_IMAGE_GRAYSCALE);
        // markedImage = cv::imread( queryImageName ); 
        // detectAndComputeQueryKeypoints( queryImage, queryKeypoints, queryDescriptors, detector );
        // matchDescriptors( queryDescriptors, trainDescriptors, matches, matcher );
        // markHomographies( markedImage, trainImages, queryKeypoints, trainKeypoints, matches);
        // cv::imwrite("out/marked.jpg", markedImage);
        // saveResultImages( queryImage, queryKeypoints, trainImages, trainKeypoints,
                    //   matches, trainImagesNames, dirToSaveResImages );
    }
    else
    {
        // check(0);
        while(ros::ok())
        {
            while(sceneID == 0) ros::spinOnce();
            // check(1);
            detectAndComputeQueryKeypoints( queryImage, queryKeypoints, queryDescriptors, detector );
            matchDescriptors( queryDescriptors, trainDescriptors, matches, matcher );
            findBestImages( trainImages, matches );
            findHomographies( queryKeypoints, trainKeypoints, matches );
            markHomographies( markedImage );
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", markedImage).toImageMsg();
            sensor_msgs::ImagePtr key_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", keyedImage).toImageMsg();
            sensor_msgs::ImagePtr grey_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", greyedImage).toImageMsg();
            std_msgs::Int32MultiArray match_msg; std_msgs::Int32 keypts_msg;
            keypts_msg.data = queryPoints;
            for(int i = 0; i < numMatches.size(); i++) match_msg.data.push_back(numMatches[i]); greyPub.publish(grey_msg);
            imgPub.publish(msg), keyPub.publish(key_msg), countPub.publish(keypts_msg), matchPub.publish(match_msg);
            // queryKeypoints.clear(), matches.clear(), numMatches.clear();
            ros::spinOnce();
            loopRate.sleep();
        }
    }

    return 0;
}