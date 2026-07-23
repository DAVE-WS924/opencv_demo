#include "feature_point_detection.h"

namespace feature_point_detection {

// 特征点检测
// 特征点检测是提取图像中具有独特性质的点，这些点通常具有旋转、缩放、光照不变性。

std::vector<cv::KeyPoint> orbFeaturePointDetection(const cv::Mat& src, int maxFeatures) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;  // ORB 同时算描述子, 这里只用关键点, 但 API 要求传
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->setMaxFeatures(maxFeatures);  // 限制数量, 按响应强度保留前 maxFeatures 个
    orb->detectAndCompute(src, cv::noArray(), keypoints, descriptors);
    return keypoints;
}

cv::Mat drawORBKeypoints(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat vis;
    // DEFAULT: 画小圆圈 (固定半径) + 十字准星, 不跟 keypoint.size 走, 避免圆圈过大糊在一起
    // 代价: 看不出尺度和方向, 但位置清晰
    cv::drawKeypoints(img, keypoints, vis, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DEFAULT);
    return vis;
}

std::vector<cv::KeyPoint> siftFeaturePointDetection(const cv::Mat& src, int maxFeatures) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;  // SIFT 同时算描述子, 这里只用关键点, 但 API 要求传
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(maxFeatures);
    sift->detectAndCompute(src, cv::noArray(), keypoints, descriptors);
    return keypoints;
}

cv::Mat drawSIFTKeypoints(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat vis;
    // DRAW_RICH_KEYPOINTS: 画圆圈 + 线段, 圆圈半径 = keypoint.size/2, 线段方向 = keypoint.angle
    // 代价: 画图慢, 特别是关键点多时, 但能看出尺度和方向
    cv::drawKeypoints(img, keypoints, vis, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return vis;
}

// ============================ 特征匹配 ============================

FeatureMatch matchORBFeatures(const cv::Mat& gray1, const cv::Mat& gray2, int maxFeatures) {
    FeatureMatch result;
    cv::Mat desc1, desc2;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(maxFeatures);
    orb->detectAndCompute(gray1, cv::noArray(), result.keypoints1, desc1);
    orb->detectAndCompute(gray2, cv::noArray(), result.keypoints2, desc2);

    if (desc1.empty() || desc2.empty()) return result;

    // BFMatcher + Hamming 距离
    //   ORB 是二进制描述子, 用 Hamming 距离 (不同 bit 的个数) 而不是 L2
    //   SIFT/SURF 是浮点描述子, 才用 L2
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    // knn=2: 每个描述子找最近 2 个匹配, 用于 ratio test
    matcher.knnMatch(desc1, desc2, knnMatches, 2);

    // Lowe's ratio test (SIFT 论文里的经典做法, 也适用于 ORB):
    //   如果 最近距离 m[0] < ratio * 次近距离 m[1], 认为是好匹配.
    //   直觉: 最近和次最近差不多近 -> 匹配有歧义, 丢掉
    //         最近明显比次最近近   -> 找到明确对应, 保留
    //   ratio 越小越严格 (0.6 严格, 0.8 宽松, 0.75 折中)
    const float ratio = 0.75f;
    for (const auto& m : knnMatches) {
        if (m.size() >= 2 && m[0].distance < ratio * m[1].distance) {
            result.goodMatches.push_back(m[0]);
        }
    }
    return result;
}

cv::Mat drawFeatureMatches(const cv::Mat& img1, const cv::Mat& img2, const FeatureMatch& m) {
    cv::Mat vis;
    // drawMatches 自动把两幅图横向并排画好, 匹配上的关键点用线连接
    //   Scalar(0,255,0)  = 匹配点 + 连接线 (绿)
    //   Scalar(0,0,255)  = 单边点 (红, 这里 NOT_DRAW_SINGLE_POINTS 不画)
    //   NOT_DRAW_SINGLE_POINTS = 不画没匹配上的点, 避免视觉干扰
    cv::drawMatches(img1, m.keypoints1, img2, m.keypoints2,
                    m.goodMatches, vis,
                    cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return vis;
}

}  // namespace feature_point_detection
