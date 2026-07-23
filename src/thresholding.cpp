#include "thresholding.h"

namespace thresholding {

cv::Mat binarize(const cv::Mat& gray, double thresh, double maxVal) {
    cv::Mat out;
    cv::threshold(gray, out, thresh, maxVal, cv::THRESH_BINARY);
    return out;
}

cv::Mat adaptiveBinarize(const cv::Mat& gray, double maxVal,
                         int method, int blockSize, double C) {
    cv::Mat out;
    cv::adaptiveThreshold(gray, out, maxVal, method, cv::THRESH_BINARY, blockSize, C);
    return out;
}
//
cv::Mat otsuBinarize(const cv::Mat& gray, double maxVal) {
    cv::Mat out;
    cv::threshold(gray, out, 0, maxVal, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return out;
}

}  // namespace thresholding
