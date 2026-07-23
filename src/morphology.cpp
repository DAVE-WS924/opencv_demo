#include "morphology.h"

namespace morphology {

cv::Mat createKernel(int shape, cv::Size ksize) {
    return cv::getStructuringElement(shape, ksize);
}

cv::Mat erodeImage(const cv::Mat& img, const cv::Mat& kernel) {
    cv::Mat out;
    cv::erode(img, out, kernel);
    return out;
}

cv::Mat dilateImage(const cv::Mat& img, const cv::Mat& kernel) {
    cv::Mat out;
    cv::dilate(img, out, kernel);
    return out;
}

cv::Mat morphOpen(const cv::Mat& img, const cv::Mat& kernel) {
    cv::Mat out;
    cv::morphologyEx(img, out, cv::MORPH_OPEN, kernel);
    return out;
}

cv::Mat morphClose(const cv::Mat& img, const cv::Mat& kernel) {
    cv::Mat out;
    cv::morphologyEx(img, out, cv::MORPH_CLOSE, kernel);
    return out;
}

cv::Mat morphGradient(const cv::Mat& img, const cv::Mat& kernel) {
    cv::Mat out;
    cv::morphologyEx(img, out, cv::MORPH_GRADIENT, kernel);
    return out;
}

}  // namespace morphology
