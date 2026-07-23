#include "feature_detection.h"

namespace feature_detection {

cv::Mat harrisCornerDetection(const cv::Mat& gray, int blockSize, int apertureSize, double k) {
    cv::Mat dst, dst_norm, dst_norm_scaled;
    cv::cornerHarris(gray, dst, blockSize, apertureSize, k);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    return dst_norm_scaled;
}

std::vector<cv::Point2f> detectHarrisCorners(const cv::Mat& gray,
                                             double threshRatio,
                                             int nmsSize,
                                             cv::Size subPixWin,
                                             int blockSize,
                                             int apertureSize,
                                             double k,
                                             int maxCorners) {
    cv::Mat dst, dst_norm;
    cv::cornerHarris(gray, dst, blockSize, apertureSize, k);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);

    double thresh = threshRatio * 255.0;

    cv::Mat dilated;
    cv::Mat nmsKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(nmsSize, nmsSize));
    cv::dilate(dst_norm, dilated, nmsKernel);
    cv::Mat localMax = (dst_norm == dilated);

    struct Corner { cv::Point2f pt; float resp; };
    std::vector<Corner> candidates;
    for (int y = 0; y < dst_norm.rows; ++y) {
        for (int x = 0; x < dst_norm.cols; ++x) {
            float resp = dst_norm.at<float>(y, x);
            if (resp > thresh && localMax.at<uchar>(y, x)) {
                candidates.push_back({cv::Point2f((float)x, (float)y), resp});
            }
        }
    }

    if (maxCorners > 0 && candidates.size() > (size_t)maxCorners) {
        std::partial_sort(candidates.begin(),
                          candidates.begin() + maxCorners,
                          candidates.end(),
                          [](const Corner& a, const Corner& b) {
                              return a.resp > b.resp;
                          });
        candidates.resize(maxCorners);
    }

    std::vector<cv::Point2f> corners;
    corners.reserve(candidates.size());
    for (const auto& c : candidates) corners.push_back(c.pt);

    if (!corners.empty()) {
        cv::TermCriteria criteria(
            cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 40, 0.001);
        cv::cornerSubPix(gray, corners, subPixWin, cv::Size(-1, -1), criteria);
    }
    return corners;
}

std::vector<cv::Point2f> shiTomasiCornerDetection(const cv::Mat& gray, int maxCorners, double qualityLevel, double minDistance, int blockSize){
    std::vector<cv::Point2f> corners;
    // 这里固定 false, 函数名已经表明是 Shi-Tomasi
    cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, 0.04);
    return corners;
}

cv::Mat drawCorners(const cv::Mat& img, const std::vector<cv::Point2f>& corners) {
    cv::Mat vis = img.clone();
    for (const auto& p : corners) {
        cv::circle(vis, p, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(vis, p, 5, cv::Scalar(0, 255, 0), 1);
    }
    return vis;
}

}  // namespace feature_detection
