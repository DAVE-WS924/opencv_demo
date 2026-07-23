#pragma once
#include <opencv2/opencv.hpp>

namespace contour_detection {

    cv::Mat detectContours(const cv::Mat& img, int mode = cv::RETR_EXTERNAL, int method = cv::CHAIN_APPROX_SIMPLE);
}  // namespace contour_detection
