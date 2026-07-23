#include "contour_detection.h"

namespace contour_detection {

    cv::Mat detectContours(const cv::Mat& img, int mode, int method){

        cv::Mat gray;
        if (img.channels() == 3) {
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = img.clone();
        }

        cv::Mat binary;
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binary, contours, hierarchy, mode, method);

        cv::Mat contourImage = cv::Mat::zeros(img.size(), CV_8UC3);
        for (size_t i = 0; i < contours.size(); ++i) {
            cv::drawContours(contourImage, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
        }

        return contourImage;
    }
}  // namespace contour_detection
