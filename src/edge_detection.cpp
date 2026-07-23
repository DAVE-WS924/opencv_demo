#include "edge_detection.h"

namespace edge_detection {

cv::Mat cannyEdge(const cv::Mat& gray, double threshold1, double threshold2) {
    cv::Mat out;
    cv::Canny(gray, out, threshold1, threshold2);
    return out;
}

cv::Mat sobelEdge(const cv::Mat& gray, double alpha, double beta) {
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_16S, 1, 0);
    cv::Sobel(gray, grad_y, CV_16S, 0, 1);
    cv::convertScaleAbs(grad_x, grad_x);
    cv::convertScaleAbs(grad_y, grad_y);
    cv::Mat dst;
    cv::addWeighted(grad_x, alpha, grad_y, beta, 0, dst);
    return dst;
}

}  // namespace edge_detection
