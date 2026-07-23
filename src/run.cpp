#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include<vector>
#include<limits>
#include<fstream>
#include<sstream>
#include<cmath>
#include<filesystem>
#include <algorithm>
#include "filter.h"
#include "morphology.h"
#include "thresholding.h"
#include "histogram.h"
#include "edge_detection.h"
#include "feature_detection.h"
#include "barcode_detection.h"
#include "contour_detection.h"
#include "feature_point_detection.h"
#include"feature_point_detection.h"

// ============================ 显示工具 ============================

// 把两张图横向拼接, 顶部加标签, 自动处理通道/尺寸差异
static cv::Mat makeSideBySide(const cv::Mat& left, const std::string& leftLabel,
                              const cv::Mat& right, const std::string& rightLabel) {
    cv::Mat leftBGR, rightBGR;
    if (left.channels() == 1)  cv::cvtColor(left,  leftBGR,  cv::COLOR_GRAY2BGR);
    else                        leftBGR  = left;
    if (right.channels() == 1) cv::cvtColor(right, rightBGR, cv::COLOR_GRAY2BGR);
    else                        rightBGR = right;

    // 统一高度 (按较小的缩, 避免放大失真)
    int h = std::min(leftBGR.rows, rightBGR.rows);
    cv::Mat leftR, rightR;
    if (leftBGR.rows != h) {
        double s = (double)h / leftBGR.rows;
        cv::resize(leftBGR, leftR, cv::Size(), s, s);
    } else leftR = leftBGR;
    if (rightBGR.rows != h) {
        double s = (double)h / rightBGR.rows;
        cv::resize(rightBGR, rightR, cv::Size(), s, s);
    } else rightR = rightBGR;

    cv::Mat combined;
    cv::hconcat(leftR, rightR, combined);

    // 顶部标签
    cv::putText(combined, leftLabel,  cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    cv::putText(combined, rightLabel, cv::Point(leftR.cols + 10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    return combined;
}

// 显示原图 + 结果并排
static void show(const std::string& name, const cv::Mat& img, const cv::Mat& result) {
    if (result.empty()) { std::cerr << name << " 结果为空\n"; return; }
    cv::Mat combined = makeSideBySide(img, "Original", result, name);
    cv::imshow(name, combined);
}

// ============================ 菜单工具 ============================

static int readChoice() {
    int choice = -1;
    if (!(std::cin >> choice)) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        return -1;
    }
    return choice;
}

static void showMainMenu() {
    std::cout << "\n===== OpenCV 图像处理菜单 =====\n"
              << "1. 滤波\n"
              << "2. 形态学\n"
              << "3. 阈值化\n"
              << "4. 直方图\n"
              << "5. 边缘检测\n"
              << "6. 特征检测\n"
              << "7. 二维码角点\n"
              << "8. 轮廓检测\n"
              << "0. 退出\n"
              << "请选择分类: ";
}

// ============================ 各分类子菜单 ============================

static void menuFilter(const cv::Mat& img) {
    while (true) {
        std::cout << "\n--- 滤波 ---\n"
                  << "1. 均值滤波\n"
                  << "2. 高斯滤波\n"
                  << "3. 中值滤波\n"
                  << "0. 返回主菜单\n"
                  << "请选择: ";
        int c = readChoice();
        switch (c) {
            case 0: return;
            case 1: show("blur",     img, filter::blur(img)); break;
            case 2: show("gaussian", img, filter::gaussianBlur(img)); break;
            case 3: show("median",   img, filter::medianBlur(img)); break;
            default: std::cout << "无效选项\n"; continue;
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

static void menuMorphology(const cv::Mat& img) {
    cv::Mat kernel = morphology::createKernel();
    while (true) {
        std::cout << "\n--- 形态学 ---\n"
                  << "1. 腐蚀\n"
                  << "2. 膨胀\n"
                  << "3. 开运算\n"
                  << "4. 闭运算\n"
                  << "5. 梯度运算\n"
                  << "0. 返回主菜单\n"
                  << "请选择: ";
        int c = readChoice();
        switch (c) {
            case 0: return;
            case 1: show("erode",    img, morphology::erodeImage(img, kernel)); break;
            case 2: show("dilate",   img, morphology::dilateImage(img, kernel)); break;
            case 3: show("open",     img, morphology::morphOpen(img, kernel)); break;
            case 4: show("close",    img, morphology::morphClose(img, kernel)); break;
            case 5: show("gradient", img, morphology::morphGradient(img, kernel)); break;
            default: std::cout << "无效选项\n"; continue;
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

static void menuThresholding(const cv::Mat& gray, const cv::Mat& img) {
    while (true) {
        std::cout << "\n--- 阈值化 ---\n"
                  << "1. 二值化\n"
                  << "2. 自适应阈值\n"
                  << "3. Otsu 阈值\n"
                  << "0. 返回主菜单\n"
                  << "请选择: ";
        int c = readChoice();
        switch (c) {
            case 0: return;
            case 1: show("binary",   img, thresholding::binarize(gray)); break;
            case 2: show("adaptive", img, thresholding::adaptiveBinarize(gray)); break;
            case 3: show("otsu",     img, thresholding::otsuBinarize(gray)); break;
            default: std::cout << "无效选项\n"; continue;
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

static void menuHistogram(const cv::Mat& gray, const cv::Mat& img) {
    while (true) {
        std::cout << "\n--- 直方图 ---\n"
                  << "1. 直方图\n"
                  << "2. 直方图均衡化\n"
                  << "0. 返回主菜单\n"
                  << "请选择: ";
        int c = readChoice();
        switch (c) {
            case 0: return;
            case 1: show("hist",      img, histogram::drawHist(histogram::computeHist(gray))); break;
            case 2: show("equalized", img, histogram::equalizeHistogram(gray)); break;
            default: std::cout << "无效选项\n"; continue;
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

static void menuEdge(const cv::Mat& gray, const cv::Mat& img) {
    while (true) {
        std::cout << "\n--- 边缘检测 ---\n"
                  << "1. Canny 边缘\n"
                  << "2. Sobel 边缘\n"
                  << "0. 返回主菜单\n"
                  << "请选择: ";
        int c = readChoice();
        switch (c) {
            case 0: return;
            case 1: show("canny", img, edge_detection::cannyEdge(gray)); break;
            case 2: show("sobel", img, edge_detection::sobelEdge(gray)); break;
            default: std::cout << "无效选项\n"; continue;
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

static void menuFeature(const cv::Mat& gray, const cv::Mat& img) {
    while (true) {
        std::cout << "\n--- 特征检测 ---\n"
                  << "1. Harris 响应图\n"
                  << "2. Harris 角点提取 (亚像素)\n"
                  << "3. Shi-Tomasi 角点提取\n"
                  <<"4. ORB 特征点提取\n"
                  <<"5.SIFT 特征点提取\n"
                  <<"6. 特征匹配 (ORB + BFMatcher, test1.jpg vs test2.jpg)\n"
                  << "0. 返回主菜单\n"
                  << "请选择: ";
        int c = readChoice();
        switch (c) {
            case 0: return;
            case 1: show("harris_response", img, feature_detection::harrisCornerDetection(gray)); break;
            case 2: {
                auto corners = feature_detection::detectHarrisCorners(gray, 0.05, 11,
                                                    cv::Size(5,5), 2, 3, 0.04, 4);
                std::cout << "检测到角点数: " << corners.size() << "\n";
                for (size_t i = 0; i < corners.size(); ++i) {
                    std::cout << "  corner[" << i << "]: ("
                              << corners[i].x << ", " << corners[i].y << ")\n";
                }
                show("harris_corners", img, feature_detection::drawCorners(img, corners));
                break;
            }
            case 3: {
                auto corners = feature_detection::shiTomasiCornerDetection(gray);
                std::cout << "检测到角点数: " << corners.size() << "\n";
                for (size_t i = 0; i < corners.size(); ++i) {
                    std::cout << "  corner[" << i << "]: ("
                              << corners[i].x << ", " << corners[i].y << ")\n";
                }
                show("shi_tomasi_corners", img, feature_detection::drawCorners(img, corners));
                break;
            }
            case 4: {
                auto kps = feature_point_detection::orbFeaturePointDetection(gray);
                std::cout << "检测到 ORB 特征点数: " << kps.size() << "\n";
                std::cout << "前 500 个特征点:\n";
                for (size_t i = 0; i < std::min(kps.size(), size_t(500)); ++i) {
                    std::cout << "  keypoint[" << i << "]: ("
                              << kps[i].pt.x << ", " << kps[i].pt.y << "), size=" << kps[i].size
                              << ", angle="
                              << kps[i].angle << "\n";
                }

                show("orb_keypoints", img, feature_point_detection::drawORBKeypoints(img, kps));
                break;
            }
            case 5: {
                auto kps = feature_point_detection::siftFeaturePointDetection(gray);
                std::cout << "检测到 SIFT 特征点数: " << kps.size() << "\n";
                std::cout << "前 500 个特征点:\n";
                for (size_t i = 0; i < std::min(kps.size(), size_t(500)); ++i) {
                    std::cout << "  keypoint[" << i << "]: ("
                              << kps[i].pt.x << ", " << kps[i].pt.y << "), size=" << kps[i].size
                              << ", angle="
                              << kps[i].angle << "\n";
                }

                show("sift_keypoints", img, feature_point_detection::drawSIFTKeypoints(img, kps));
                break;
            }
            case 6: {
                // 演示: 加载 test1.jpg 和 test2.jpg, 看 ORB 能匹配上多少特征点
                std::string path1 = "/home/david/camera_server_src/opencv_train/demo/test1.jpg";
                std::string path2 = "/home/david/camera_server_src/opencv_train/demo/test2.jpg";
                cv::Mat img1 = cv::imread(path1);
                cv::Mat img2 = cv::imread(path2);
                if (img1.empty() || img2.empty()) {
                    std::cerr << "无法读取图片: "
                              << (img1.empty() ? path1 + " " : "")
                              << (img2.empty() ? path2 : "") << "\n";
                    break;
                }
                cv::Mat gray1, gray2;
                cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
                cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

                auto m = feature_point_detection::matchORBFeatures(gray1, gray2);
                std::cout << "图1特征点: " << m.keypoints1.size()
                          << ", 图2特征点: " << m.keypoints2.size()
                          << ", 好匹配数: " << m.goodMatches.size() << "\n";

                // drawFeatureMatches 已经把两幅图并排画了, 直接 imshow, 不走 show() 助手
                cv::Mat vis = feature_point_detection::drawFeatureMatches(img1, img2, m);
                cv::imshow("feature_matches", vis);
                break;
            }
            default: std::cout << "无效选项\n"; continue;
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

static void menuBarcode() {
    while (true) {
        std::cout << "\n--- 二维码角点 ---\n"
                  << "1. 交互式查看 (YOLO+Harris)\n"
                  << "2. 批量保存到 data_out/ (JSON+图)\n"
                  << "0. 返回主菜单\n"
                  << "请选择: ";
        int c = readChoice();
        switch (c) {
            case 0: return;
            case 1: {
                std::string dataDir = "/home/david/camera_server_src/opencv_train/demo/data";
                barcode_detection::runQrCornerBatch(dataDir);
                break;
            }
            case 2: {
                std::string inDir  = "/home/david/camera_server_src/opencv_train/demo/data_320sz";
                std::string outDir = "/home/david/camera_server_src/opencv_train/demo/output/data_out_320_fp";
                barcode_detection::saveQrCornerBatch(inDir, outDir);
                break;
            }
            default: std::cout << "无效选项\n"; continue;
        }
        // 这两项自己控制 imshow/waitKey, 不额外 waitKey
    }
}

static void menuContour(const cv::Mat& img) {
    while (true) {
        std::cout << "\n--- 轮廓检测 ---\n"
                  << "1. 轮廓检测\n"
                  << "0. 返回主菜单\n"
                  << "请选择: ";
        int c = readChoice();
        switch (c) {
            case 0: return;
            case 1: show("contours", img, contour_detection::detectContours(img)); break;
            default: std::cout << "无效选项\n"; continue;
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

// ============================ 主流程 ============================

int main() {
    std::string path = "/home/david/camera_server_src/opencv_train/test.jpg";
    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "无法读取图片: " << path << "\n";
        return -1;
    }

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    while (true) {
        showMainMenu();
        int c = readChoice();
        switch (c) {
            case 0:  cv::destroyAllWindows(); return 0;
            case 1:  menuFilter(img); break;
            case 2:  menuMorphology(img); break;
            case 3:  menuThresholding(gray, img); break;
            case 4:  menuHistogram(gray, img); break;
            case 5:  menuEdge(gray, img); break;
            case 6:  menuFeature(gray, img); break;
            case 7:  menuBarcode(); break;
            case 8:  menuContour(img); break;
            default: std::cout << "无效选项\n"; continue;
        }
    }
    return 0;
}
