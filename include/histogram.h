#pragma once
#include<opencv2/opencv.hpp>

namespace histogram {

    // ============================ 直方图 ============================

// 计算单通道图像的直方图
cv::Mat computeHist(const cv::Mat& gray, int histSize = 256);

// 直方图均衡化: 增强对比度, 使亮度分布更均匀
cv::Mat equalizeHistogram(const cv::Mat& gray);

// 比较两个单通道图像的直方图相似性
double compareHistograms(const cv::Mat& gray1, const cv::Mat& gray2,
                         int method = cv::HISTCMP_CORREL);

// 把直方图绘制成可视化图像
cv::Mat drawHist(const cv::Mat& hist, int histSize = 256,
                 int histW = 512, int histH = 400);
}