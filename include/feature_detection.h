#pragma once
#include <opencv2/opencv.hpp>

namespace feature_detection {
// Harris角点检测: 计算响应图
cv::Mat harrisCornerDetection(const cv::Mat& gray, int blockSize = 2, int apertureSize = 3, double k = 0.04) ;

// Harris角点提取: 阈值筛选 + 非极大值抑制 + top-N + 亚像素细化
// 返回亚像素精度的角点坐标 (0.05~0.1 px)
//   threshRatio : 阈值占最大响应的比例 (0.01~0.1)
//   nmsSize     : 非极大值抑制邻域, 角点间距需大于此值
//   subPixWin   : 亚像素搜索窗口, 太大会受相邻边干扰
//   maxCorners  : 最多保留的角点数, 按响应值从强到弱排序; <=0 表示不限制
std::vector<cv::Point2f> detectHarrisCorners(const cv::Mat& gray,
                                             double threshRatio = 0.01,
                                             int nmsSize = 5,
                                             cv::Size subPixWin = cv::Size(5, 5),
                                             int blockSize = 2,
                                             int apertureSize = 3,
                                             double k = 0.04,
                                             int maxCorners = 0);
//shi-tomasi角点检测: 基于 goodFeaturesToTrack (useHarrisDetector=false), 用最小特征值作为角点强度
//   maxCorners   : 最多保留的角点数 (按响应强度从强到弱排序)
//   qualityLevel : 角点质量阈值, 低于 max(响应)*qualityLevel 的点被丢弃 (典型 0.01~0.05)
//   minDistance  : 角点之间的最小距离 (像素), 避免角点聚堆
//   blockSize    : 邻域大小, 用于计算自相关矩阵
std::vector<cv::Point2f> shiTomasiCornerDetection(const cv::Mat& gray,
                                                   int maxCorners = 100,
                                                   double qualityLevel = 0.01,
                                                   double minDistance = 10,
                                                   int blockSize = 3);
// 在图上画出角点, 返回可视化图像
cv::Mat drawCorners(const cv::Mat& img, const std::vector<cv::Point2f>& corners);
}