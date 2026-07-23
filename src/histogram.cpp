#include "histogram.h"

namespace histogram {

cv::Mat computeHist(const cv::Mat& gray, int histSize) {
    cv::Mat hist;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    return hist;
}

cv::Mat equalizeHistogram(const cv::Mat& gray) {
    cv::Mat out;
    cv::equalizeHist(gray, out);
    return out;
}

double compareHistograms(const cv::Mat& gray1, const cv::Mat& gray2, int method) {
    cv::Mat h1 = computeHist(gray1);
    cv::Mat h2 = computeHist(gray2);
    return cv::compareHist(h1, h2, method);
}

cv::Mat drawHist(const cv::Mat& hist, int histSize, int histW, int histH) {
    int binW = cvRound((double)histW / histSize);
    cv::Mat histImage(histH, histW, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat normalized;
    cv::normalize(hist, normalized, 0, histImage.rows, cv::NORM_MINMAX);
    for (int i = 1; i < histSize; i++) {
        cv::line(histImage,
            cv::Point(binW * (i - 1), histH - cvRound(normalized.at<float>(i - 1))),
            cv::Point(binW * i,       histH - cvRound(normalized.at<float>(i))),
            cv::Scalar(255, 255, 255), 2);
    }
    return histImage;
}

}  // namespace histogram
