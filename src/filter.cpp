#include "filter.h"

namespace filter {

cv::Mat blur(const cv::Mat& img, cv::Size ksize) {
    cv::Mat out;
    cv::blur(img, out, ksize);
    return out;
}

cv::Mat gaussianBlur(const cv::Mat& img, cv::Size ksize,
                     double sigmaX, double sigmaY) {
    cv::Mat out;
    cv::GaussianBlur(img, out, ksize, sigmaX, sigmaY);
    return out;
}

cv::Mat medianBlur(const cv::Mat& img, int ksize) {
    cv::Mat out;
    cv::medianBlur(img, out, ksize);
    return out;
}

}  // namespace filter
