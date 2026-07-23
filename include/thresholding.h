#pragma once
#include <opencv2/opencv.hpp>

namespace thresholding {

cv::Mat binarize(const cv::Mat& gray, double thresh = 150, double maxVal = 255);
cv::Mat adaptiveBinarize(const cv::Mat& gray, double maxVal = 255,
                         int method = cv::ADAPTIVE_THRESH_MEAN_C,
                         int blockSize = 11, double C = 2);
cv::Mat otsuBinarize(const cv::Mat& gray, double maxVal = 255);

}  // namespace thresholding
