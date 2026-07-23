#pragma once
#include <opencv2/opencv.hpp>

namespace filter {

// 均值滤波: 将像素替换为邻域内的均值。作用: 去除噪声, 图像变模糊
cv::Mat blur(const cv::Mat& img, cv::Size ksize = cv::Size(5, 5));

// 高斯滤波: 基于高斯函数的加权平均。作用: 去除噪声, 图像变模糊
cv::Mat gaussianBlur(const cv::Mat& img, cv::Size ksize = cv::Size(5, 5),
                     double sigmaX = 0, double sigmaY = 0);

// 中值滤波: 非线性, 像素取邻域中值。作用: 去除椒盐噪声
cv::Mat medianBlur(const cv::Mat& img, int ksize = 5);

}  // namespace filter
