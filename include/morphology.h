#pragma once
#include <opencv2/opencv.hpp>

namespace morphology {

cv::Mat createKernel(int shape = cv::MORPH_RECT, cv::Size ksize = cv::Size(5, 5));
cv::Mat erodeImage(const cv::Mat& img, const cv::Mat& kernel);
cv::Mat dilateImage(const cv::Mat& img, const cv::Mat& kernel);
cv::Mat morphOpen(const cv::Mat& img, const cv::Mat& kernel);
cv::Mat morphClose(const cv::Mat& img, const cv::Mat& kernel);
cv::Mat morphGradient(const cv::Mat& img, const cv::Mat& kernel);

}  // namespace morphology
