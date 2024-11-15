#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <string>
#include "project.h"
#include <image_transport/image_transport.h>
#include <thread>  // 需要包含此头文件
#include <memory>


std::string ext="/media/zzy/T7/ship_c_data/lidar-camera/extrinsic.txt";

double fx=1848.058547;
double fy=1847.112001;
double cx=1266.409969;
double cy=749.832821;
double k1=0.03743589;
double k2=-0.05156697;
double p1=0.00065813;
double p2=0.00032850;
double k3=0.0;

class ImageCloudProcessor
{
public:
    ImageCloudProcessor(ros::NodeHandle& nh)
        : it_(nh)
    {
        // 初始化图像发布器
        image_pub_ = it_.advertise("/processed_image", 1);

        // 加载图像和点云文件夹
        loadFiles("/media/zzy/T7/ship_c_data/lidar-camera/camera/camera", "/media/zzy/T7/ship_c_data/lidar-camera/2024-05-15-16-58-27_filter/2024-05-15-16-58-27_filter");

        // 使用异步定时器，定期调用处理函数（提高频率到20Hz）
        timer_ = nh.createTimer(ros::Duration(0.1), &ImageCloudProcessor::processNextImage, this);
    }

private:
    void loadFiles(const std::string& image_folder, const std::string& pcd_folder)
    {
        // 加载图像和点云文件路径
        for (const auto& entry : std::filesystem::directory_iterator(image_folder))
        {
            if (entry.path().extension() == ".jpg")
            {
                image_files_.push_back(entry.path().string());
            }
        }

        for (const auto& entry : std::filesystem::directory_iterator(pcd_folder))
        {
            if (entry.path().extension() == ".pcd")
            {
                pcd_files_.push_back(entry.path().string());
            }
        }

        // 排序文件，根据时间戳排序
        std::sort(image_files_.begin(), image_files_.end());
        std::sort(pcd_files_.begin(), pcd_files_.end());
    }

    void processNextImage(const ros::TimerEvent&)
    {
        if (image_idx_ >= image_files_.size()) return;

        std::string image_file = image_files_[image_idx_];
        std::string closest_pcd_file = findClosestPcdFile(image_file);

        // 使用多线程处理图像和点云数据
        std::thread processing_thread(&ImageCloudProcessor::processImageAndPublish, this, image_file, closest_pcd_file);
        processing_thread.detach();  // 将线程分离，后台执行

        // 增加图像索引
        image_idx_++;
    }

    void processImageAndPublish(const std::string& image_file, const std::string& pcd_file)
    {
        // 读取图像
        cv::Mat input_image = cv::imread(image_file);

        // 读取点云
        pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *lidar_cloud);

        // 调用 Proj2Img 函数获取投影图
        cv::Mat projected_image = Proj2Img(input_image, lidar_cloud, 1, ext, fx, fy, cx, cy, k1, k2, p1, p2, k3, false);

        // 转换为 ROS 消息并发布
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", projected_image).toImageMsg();
        image_pub_.publish(msg);
    }

    std::string findClosestPcdFile(const std::string& image_file)
    {
        double img_timestamp = std::stod(image_file.substr(image_file.find_last_of("/") + 1, 13));

        double closest_diff = std::numeric_limits<double>::max();
        std::string closest_pcd_file;
        for (const auto& pcd_file : pcd_files_)
        {
            double pcd_timestamp = std::stod(pcd_file.substr(pcd_file.find_last_of("/") + 1, 13));
            double diff = std::abs(img_timestamp - pcd_timestamp);
            if (diff < closest_diff)
            {
                closest_diff = diff;
                closest_pcd_file = pcd_file;
            }
        }

        return closest_pcd_file;
    }

    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Publisher image_pub_;
    ros::Timer timer_;
    size_t image_idx_ = 0;

    std::vector<std::string> image_files_;
    std::vector<std::string> pcd_files_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_cloud_processor");
    ros::NodeHandle nh;
    ImageCloudProcessor processor(nh);

    // 使用 MultiThreadedSpinner 代替 AsyncSpinner
    ros::MultiThreadedSpinner spinner(4); // 使用4个线程
    spinner.spin();

    return 0;
}
