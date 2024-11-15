#pragma once
#include "common.h"
//#include"calculate_pose.h"
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <unordered_map>
#include <pcl/point_types.h>
#include <cmath>
////////////////////////////////////////////////
#include <pcl/point_types.h>
#include <pcl/common/common.h>
using namespace std;
enum ProjectionType { DEPTH, INTENSITY, BOTH };

double min_depth_ = 1.5;
double max_depth_ = 10;

uchar Monochrome(int R)
{
    // 将R值映射到0-255范围内
    return static_cast<uchar>(R);
}


cv::Vec3b FalseColor(int R)
{
    cv::Vec3b color;
    switch (R / 64)
    {
    case 0:
        color = cv::Vec3b(0, 4 * R, 255);
        break;
    case 1:
        color = cv::Vec3b(0, 255, 510 - 4 * R);
        break;
    case 2:
        color = cv::Vec3b(4 * R - 510, 255, 0);
        break;
    case 3:
        color = cv::Vec3b(255, 1020 - 4 * R, 0);
        break;
    }
    return color;
}

cv::Mat Proj2Img(cv::Mat InputImage, pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_cloud,const int density, string exfile, double fx_ ,
double fy_ ,
double cx_ ,
double cy_,
double k1_ ,
double k2_,
double p1_,
double p2_,
double k3_ ,bool iffisheye)
{
    ifstream fin(exfile, ios::in);
    Eigen::Matrix3d rotation;
    Eigen::Vector3d transation;
    fin >> rotation(0, 0); fin >> rotation(0, 1); fin >> rotation(0, 2); fin >> transation(0);
    fin >> rotation(1, 0); fin >> rotation(1, 1); fin >> rotation(1, 2); fin >> transation(1);
    fin >> rotation(2, 0); fin >> rotation(2, 1); fin >> rotation(2, 2); fin >> transation(2);
    cv::Mat k1 = cv::Mat::zeros(3, 3, CV_32F); cv::Mat k2 = cv::Mat::zeros(3, 3, CV_32F); cv::Mat c1 = cv::Mat::zeros(4, 1, CV_32F);
    k1.at<float>(0, 0) = fx_; k1.at<float>(0, 1) = 0; k1.at<float>(0, 2) = cx_;
    k1.at<float>(1, 0) = 0; k1.at<float>(1, 1) = fy_; k1.at<float>(1, 2) = cy_;
    k1.at<float>(2, 0) = 0; k1.at<float>(2, 1) = 0; k1.at<float>(2, 2) = 1;
    k2.at<float>(0, 0) = fx_ * 0.8; k2.at<float>(0, 1) = 0; k2.at<float>(0, 2) = cx_;
    k2.at<float>(1, 0) = 0; k2.at<float>(1, 1) = fy_ * 0.8; k2.at<float>(1, 2) = cy_;
    k2.at<float>(2, 0) = 0; k2.at<float>(2, 1) = 0; k2.at<float>(2, 2) = 1;
    c1.at<float>(0) = k1_; c1.at<float>(1) = k2_; c1.at<float>(2) = p1_; c1.at<float>(3) = p2_; //c1.at<float>(4) = k3_;
    cv::Mat ImageProjected;
    if (iffisheye)
    {
        cv::fisheye::undistortImage(InputImage, ImageProjected, k1, c1, k2);
        ImageProjected = InputImage;
        fx_ = k2.at<float>(0, 0);
        cx_ = k2.at<float>(0, 2);
        fy_ = k2.at<float>(1, 1);
        cy_ = k2.at<float>(1, 2);
    }
    if (!iffisheye)
    {
        cv::undistort(InputImage, ImageProjected, k1, c1, k1);
    }
    
    k1_ = 0;
    k2_ = 0;
    p1_ = 0;
    p2_ = 0;
    k3_ = 0;
    double depthMax = 0;
   
    //////////////////////////////////////////////////////////////////////////////

    std::vector<cv::Point3f> pts_3d;
    for (size_t i = 0; i < lidar_cloud->size(); i += density)
    {
        pcl::PointXYZI point_3d = lidar_cloud->points[i];
        pts_3d.emplace_back(cv::Point3f(point_3d.x, point_3d.y, point_3d.z));
        double depth = point_3d.x * point_3d.x + point_3d.y * point_3d.y + point_3d.z * point_3d.z;
        depth = sqrt(depth);
        if (depthMax < depth&& depth > min_depth_ && depth < max_depth_)depthMax = depth;
    }
    std::vector<cv::Point2f> pts_2d;

    for (int i = 0; i < pts_3d.size(); i++)
    {
        Eigen::Vector3d pts_l;
        pts_l(0, 0) = pts_3d[i].x; pts_l(1, 0) = pts_3d[i].y; pts_l(2, 0) = pts_3d[i].z;
        Eigen::Vector3d pts_c;
        pts_c = rotation * pts_l +transation;
        double check = pts_c(2,0);
        pts_c(0, 0) = pts_c(0, 0) / pts_c(2, 0);
        pts_c(1, 0) = pts_c(1, 0) / pts_c(2, 0);
        pts_c(2, 0) = 1;
        double u_d; double v_d;
        if (iffisheye)
        {
            cv::Mat distCoeffs = (cv::Mat_<double>(4, 1) << c1.at<float>(0,0), c1.at<float>(1, 0), c1.at<float>(2, 0), c1.at<float>(3, 0));
            cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
                fx_, 0, cx_,
                0, fy_, cy_,
                0, 0, 1);
            cv::Mat rvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
            cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
            std::vector<cv::Point3f> objectPoints;  std::vector<cv::Point2f> imagePoints;
            objectPoints.push_back(cv::Point3f(pts_c(0, 0), pts_c(1, 0), 1));
            cv::fisheye::projectPoints(objectPoints, imagePoints, rvec, tvec, cameraMatrix, distCoeffs);
            u_d = imagePoints[0].x;
            v_d = imagePoints[0].y;
        }
        if (!iffisheye)
        {
            double u = pts_c(0, 0);
            double v = pts_c(1, 0);
            double r = u * u + v * v;
            u_d = (1 + k1_ * r + k2_ * r * r + k3_ * r * r * r) * u + (2 * p1_ * u * v + p2_ * (r + 2 * u * u));
            v_d = (1 + k1_ * r + k2_ * r * r + k3_ * r * r * r) * v + (p1_ * (r + 2 * v * v) + 2 * p2_ * u * v);
            u_d = fx_ * u_d + cx_;
            v_d = fy_ * v_d + cy_;
        }
        
        if (check < 0)
        {
            pts_2d.push_back(cv::Point2f(100000, 100000));
        }
        else
        {
            pts_2d.push_back(cv::Point2f(u_d, v_d));
        }
        
    }
   
    //undistort(image_before, image_, k1, c1, k2);
    int ImageHeight = ImageProjected.rows;
    int ImageWidth = ImageProjected.cols;

    for (size_t i = 0; i < pts_2d.size(); ++i)
    {
        cv::Point2f point_2d = pts_2d[i];
        double depth = pts_3d[i].x * pts_3d[i].x + pts_3d[i].y * pts_3d[i].y + pts_3d[i].z * pts_3d[i].z;
        depth = sqrt(depth);
        if (depth > min_depth_ && depth < max_depth_)
        {
            cv::Vec3b Color = FalseColor((uchar)(255 * depth / depthMax));//depthMax
            /*cv::Vec3b Color = cv::Vec3b(Monochrome((uchar)(255 * depth / depthMax)),
                Monochrome((uchar)(255 * depth / depthMax)),
                Monochrome((uchar)(255 * depth / depthMax)));*/
            if (point_2d.x >= 1 && point_2d.x < ImageWidth - 1 && point_2d.y >= 1 && point_2d.y < ImageHeight - 1)
            {
                cv::circle(ImageProjected, cv::Point2f((int)pts_2d[i].x, (int)pts_2d[i].y), 1, Color, -1);
                /*ImageProjected.at<cv::Vec3b>((int)pts_2d[i].y, (int)pts_2d[i].x) = Color;
                ImageProjected.at<cv::Vec3b>((int)pts_2d[i].y - 1, (int)pts_2d[i].x) = Color;
                ImageProjected.at<cv::Vec3b>((int)pts_2d[i].y, (int)pts_2d[i].x - 1) = Color;
                ImageProjected.at<cv::Vec3b>((int)pts_2d[i].y - 1, (int)pts_2d[i].x - 1) = Color;
                ImageProjected.at<cv::Vec3b>((int)pts_2d[i].y + 1, (int)pts_2d[i].x) = Color;
                ImageProjected.at<cv::Vec3b>((int)pts_2d[i].y, (int)pts_2d[i].x + 1) = Color;
                ImageProjected.at<cv::Vec3b>((int)pts_2d[i].y + 1, (int)pts_2d[i].x + 1) = Color;
                ImageProjected.at<cv::Vec3b>((int)pts_2d[i].y + 1, (int)pts_2d[i].x - 1) = Color;
                ImageProjected.at<cv::Vec3b>((int)pts_2d[i].y - 1, (int)pts_2d[i].x + 1) = Color;*/
            }
        }
    }
    return ImageProjected;
}


double GetRad(double angle)
{
    return angle * 3.14159265358 / 180;
}