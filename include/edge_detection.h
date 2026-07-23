#pragma once
#include <opencv2/opencv.hpp>

// ============================ 边缘检测 ============================

namespace edge_detection {
// Canny 边缘检测: 多阶段方法, 对噪声有一定鲁棒性
cv::Mat cannyEdge(const cv::Mat& gray, double threshold1 = 100, double threshold2 = 200);

// Sobel 边缘检测: 基于梯度的方法, 合并水平与垂直方向
cv::Mat sobelEdge(const cv::Mat& gray, double alpha = 0.5, double beta = 0.5);
}
    