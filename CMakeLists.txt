cmake_minimum_required(VERSION 2.8.3)
project(feature_detector)

find_package(catkin REQUIRED)
find_package(catkin_simple REQUIRED)
catkin_simple()

add_definitions(-std=c++11)
set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

cs_add_executable(flann_sift src/flann_sift.cxx)
target_link_libraries(flann_sift ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} opencv_xfeatures2d)

cs_add_executable(sift src/sift.cxx)
target_link_libraries(sift ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} opencv_xfeatures2d)

cs_add_executable(knn_sift src/knn_sift.cxx)
target_link_libraries(knn_sift ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} opencv_xfeatures2d)

cs_add_executable(sample src/sample.cxx)
target_link_libraries(sample ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} opencv_xfeatures2d)

cs_install()
cs_export()
