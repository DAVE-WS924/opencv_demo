#pragma once
#include <opencv2/opencv.hpp>

namespace feature_point_detection {

// ORB 特征点检测: 速度快, 兼具旋转不变性和尺度不变性
// 返回 KeyPoint 列表 (含位置、尺度、方向、响应强度)
//   src          : 单通道灰度图 (ORB 内部用 FAST 检测 + BRIEF 描述子)
//   maxFeatures  : 最多保留的特征点数, 按响应强度从强到弱排序 (默认 100)
//                  ORB 默认 500 个, 太多了画图会糊成一片
std::vector<cv::KeyPoint> orbFeaturePointDetection(const cv::Mat& src, int maxFeatures = 1000);

// 把 ORB 关键点画在图上, 返回可视化图像
// 用 DRAW_RICH_KEYPOINTS, 画出关键点的尺度 (圆圈大小) 和方向 (线段)
cv::Mat drawORBKeypoints(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints);

//Sift特征点检测: 速度慢, 兼具旋转不变性和尺度不变性
// 返回 KeyPoint 列表 (含位置、尺度、方向、响应强度)
std::vector<cv::KeyPoint> siftFeaturePointDetection(const cv::Mat& src, int maxFeatures = 1000);

cv::Mat drawSIFTKeypoints(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints);

// ============================ 特征匹配 ============================

// 特征匹配结果
struct FeatureMatch {
    std::vector<cv::KeyPoint> keypoints1;   // 图 1 的关键点
    std::vector<cv::KeyPoint> keypoints2;   // 图 2 的关键点
    std::vector<cv::DMatch> goodMatches;    // 通过 ratio test 筛出的好匹配
};

// 用 ORB + BFMatcher + Lowe ratio test 匹配两幅图的特征点
//   gray1, gray2 : 两幅单通道灰度图
//   maxFeatures  : 每幅图最多检测的特征点数 (默认 500)
// 返回: 匹配结果, 包含两图的关键点和筛选后的好匹配
FeatureMatch matchORBFeatures(const cv::Mat& gray1, const cv::Mat& gray2,
                              int maxFeatures = 500);

// 画匹配结果: 两幅图并排, 匹配上的点用线连接
cv::Mat drawFeatureMatches(const cv::Mat& img1, const cv::Mat& img2,
                           const FeatureMatch& m);

}  // namespace feature_point_detection
