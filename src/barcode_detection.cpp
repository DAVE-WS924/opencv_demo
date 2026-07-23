#include "barcode_detection.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace barcode_detection {

std::vector<YoloMark> readYoloMarks(const std::string& txtPath) {
    std::vector<YoloMark> marks;
    std::ifstream f(txtPath);
    if (!f) return marks;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        YoloMark m{0, 0, 0, 0, 0, 0};
        if (ss >> m.cls >> m.cx >> m.cy >> m.w >> m.h) {
            ss >> m.conf;
            marks.push_back(m);
        }
    }
    return marks;
}

std::vector<CoarseCorner> marksToCoarse(const std::vector<YoloMark>& marks,
                                         int W, int H) {
    std::vector<CoarseCorner> out;
    out.reserve(marks.size());
    for (const auto& m : marks) {
        CoarseCorner c;
        c.pt  = cv::Point2f((float)(m.cx * W), (float)(m.cy * H));
        c.wPx = (float)(m.w * W);
        c.hPx = (float)(m.h * H);
        out.push_back(c);
    }
    return out;
}

// 用 Harris 角点响应把 YOLO 粗定位的角点精定位到亚像素精度
//
// 背景: YOLO 检测出的角点坐标 (coarse) 通常有 ±几像素的误差, 不能直接用于
//       几何测量 (如二维码解码、单应矩阵求解). 这个函数在每个粗角点附近开一个小窗口,
//       找到窗口内 Harris 响应最强的像素 (整数级精定位), 再用亚像素细化到 0.05~0.1 px.
//
// 两阶段设计的动机:
//   阶段 1 (整数级): Harris 响应图是离散的, 像素级最亮点的位置就是真实角点附近,
//                    但受限于像素网格, 精度只能到 1 px. 这一步纠正 YOLO 的偏差,
//                    把粗角点对齐到最近的局部最强角点像素.
//   阶段 2 (亚像素): 在阶段 1 找到的像素周围用二次曲面拟合 Harris 响应, 求曲面极值点,
//                    得到亚像素坐标. 最终精度可达 0.05~0.1 px.
//
// 参数:
//   gray         : 单通道灰度图
//   coarse       : YOLO 给出的粗角点列表 (归一化坐标已转成像素)
//   searchRadius : 阶段 1 在粗角点周围多大半径内搜索最强响应像素
//                  太小 -> YOLO 误差大时会跳过真实角点
//                  太大 -> 可能抓到相邻角 (二维码 4 角离得近时尤其危险)
//   winSize      : 阶段 2 亚像素细化的搜索窗口半边长 (实际窗口 = 2*winSize+1)
//                  太大 -> 受相邻边缘干扰, 拟合偏移
//                  太小 -> 没足够样本拟合二次曲面, 细化失败
//   blockSize    : Harris 检测器邻域大小 (典型 2~5)
//   apertureSize : Sobel 算子孔径 (典型 3/5/7)
//   k            : Harris 敏感系数 (典型 0.04~0.06)
std::vector<cv::Point2f> refineCornersWithHarris(
    const cv::Mat& gray,
    const std::vector<CoarseCorner>& coarse,
    int searchRadius,
    cv::Size winSize,
    int blockSize, int apertureSize, double k) {

    // 整张图算一次 Harris 响应, 后面所有粗角点共用, 避免重复计算
    cv::Mat resp, respNorm;
    cv::cornerHarris(gray, resp, blockSize, apertureSize, k);
    // 归一化到 0~255. 不影响 argmax (找最亮点的相对结果), 只是为了数值范围稳定 
    // respNorm：这是 Harris 响应的归一化版本
    cv::normalize(resp, respNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1);

    std::vector<cv::Point2f> refined;
    refined.reserve(coarse.size());

    // ===== 阶段 1: 整数级精定位 - 每个粗角点附近找响应最强的像素 =====
    for (const auto& c : coarse) {
        const cv::Point2f& p = c.pt;
        // 以粗角点为中心、边长 2*searchRadius+1 的正方形搜索窗口（小窗口找最大值）
        // 钳到图像边界内, 防止越界访问
        int x0 = std::max(0, (int)std::round(p.x) - searchRadius);
        int y0 = std::max(0, (int)std::round(p.y) - searchRadius);
        int x1 = std::min(respNorm.cols, (int)std::round(p.x) + searchRadius + 1);
        int y1 = std::min(respNorm.rows, (int)std::round(p.y) + searchRadius + 1);

        // 在窗口内扫描, 记录响应最大的像素 (局部极大值)（小窗口找最大值）
        float best = -1.f; cv::Point2f bestPt = p;
        for (int y = y0; y < y1; ++y) {
            // 用行指针避免 at<>() 每次都做边界检查, 热点循环里能快不少
            const float* row = respNorm.ptr<float>(y);
            for (int x = x0; x < x1; ++x) {
                if (row[x] > best) { best = row[x]; bestPt = cv::Point2f((float)x, (float)y); }
            }
        }
        refined.push_back(bestPt);
    }

    // ===== 阶段 2: 亚像素细化 - 一次性对全部角点做 =====
    // 原理: 在每个角点附近的小窗口内, 把 Harris 响应当成二次曲面拟合,
    //       求曲面极值点得到亚像素坐标. 整批一起调用比逐个调用快 (内部可并行).
    // 终止条件: 40 次迭代 OR 位置变化 < 0.001 px (哪个先到就停)
    if (!refined.empty()) {
        cv::TermCriteria tc(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 40, 0.001);
        // winSize        : 主搜索窗口半边长 (用来拟合二次曲面的区域)
        // cv::Size(-1,-1): dead zone 半边长, (-1,-1) 表示无 dead zone
        //                  (dead zone 是窗口中心不参与拟合的区域, 用于避免自相关矩阵奇异;
        //                   一般不需要, 用 -1 默认)
        cv::cornerSubPix(gray, refined, winSize, cv::Size(-1, -1), tc);
    }
    return refined;
}

cv::Mat drawQrCorners(const cv::Mat& img,
                      const std::vector<CoarseCorner>& coarse,
                      const std::vector<cv::Point2f>& refined) {
    cv::Mat vis = img.clone();
    for (size_t i = 0; i < coarse.size(); ++i) {
        const auto& cp = coarse[i].pt;
        const auto& rp = refined[i];
        cv::circle(vis, cp, 8, cv::Scalar(0, 165, 255), 2);
        cv::putText(vis, "Y" + std::to_string(i), cp + cv::Point2f(-3, -12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 165, 255), 1);
        cv::circle(vis, rp, 3, cv::Scalar(0, 0, 255), -1);
        cv::putText(vis, "H" + std::to_string(i), rp + cv::Point2f(6, 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
        cv::arrowedLine(vis, cp, rp, cv::Scalar(200, 200, 200), 1, 8, 0, 0.3);
    }

    const int roiSize = 30;
    const int zoom = 4;
    const int cellW = roiSize * zoom;
    if (!coarse.empty()) {
        int n = (int)coarse.size();
        cv::Mat zoomPanel(cellW, cellW * n, img.type(), cv::Scalar(40, 40, 40));
        for (int i = 0; i < n; ++i) {
            int cx = (int)std::round(coarse[i].pt.x);
            int cy = (int)std::round(coarse[i].pt.y);
            int x0 = std::max(0, cx - roiSize / 2);
            int y0 = std::max(0, cy - roiSize / 2);
            int x1 = std::min(img.cols, x0 + roiSize);
            int y1 = std::min(img.rows, y0 + roiSize);
            cv::Rect roiRect(x0, y0, x1 - x0, y1 - y0);
            cv::Mat roi = img(roiRect);

            cv::Mat roiZoom;
            cv::resize(roi, roiZoom, cv::Size(roiRect.width * zoom, roiRect.height * zoom),
                       0, 0, cv::INTER_NEAREST);

            auto mapPt = [&](const cv::Point2f& p) {
                return cv::Point2f((p.x - x0) * zoom, (p.y - y0) * zoom);
            };
            cv::circle(roiZoom, mapPt(coarse[i].pt), 8, cv::Scalar(0, 165, 255), 2);
            cv::circle(roiZoom, mapPt(refined[i]), 3, cv::Scalar(0, 0, 255), -1);
            cv::arrowedLine(roiZoom, mapPt(coarse[i].pt), mapPt(refined[i]),
                            cv::Scalar(200, 200, 200), 1, 8, 0, 0.3);
            cv::putText(roiZoom, "corner " + std::to_string(i), cv::Point(5, 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            cv::Mat target = zoomPanel(cv::Rect(i * cellW, 0, roiZoom.cols, roiZoom.rows));
            roiZoom.copyTo(target);
        }

        int panelW = std::min(zoomPanel.cols, vis.cols);
        int panelH = std::min(zoomPanel.rows, vis.rows / 2);
        cv::Rect roiOnVis(vis.cols - panelW, 0, panelW, panelH);
        cv::Mat scaledPanel;
        cv::resize(zoomPanel, scaledPanel, roiOnVis.size(), 0, 0, cv::INTER_NEAREST);
        cv::Mat visRoi = vis(roiOnVis);
        cv::addWeighted(scaledPanel, 1.0, visRoi, 0.0, 0, visRoi);
        cv::rectangle(vis, roiOnVis, cv::Scalar(255, 255, 255), 1);
        cv::putText(vis, "4x zoom", cv::Point(roiOnVis.x + 5, roiOnVis.y + 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    }
    return vis;
}

bool processQrImage(const std::string& imgPath,
                    cv::Mat& imgOut,
                    std::vector<CoarseCorner>& coarseOut,
                    std::vector<cv::Point2f>& refinedOut) {
    imgOut = cv::imread(imgPath);
    if (imgOut.empty()) return false;

    std::string txtPath = imgPath.substr(0, imgPath.size() - 4) + ".txt";
    auto marks = readYoloMarks(txtPath);
    if (marks.empty()) return false;

    coarseOut = marksToCoarse(marks, imgOut.cols, imgOut.rows);

    cv::Mat gray;
    cv::cvtColor(imgOut, gray, cv::COLOR_BGR2GRAY);
    refinedOut = refineCornersWithHarris(gray, coarseOut);
    return true;
}

void runQrCornerBatch(const std::string& dir) {
    namespace fs = std::filesystem;
    if (!fs::exists(dir)) {
        std::cerr << "目录不存在: " << dir << "\n";
        return;
    }
    std::vector<std::string> imgs;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() == ".jpg") imgs.push_back(e.path().string());
    }
    std::sort(imgs.begin(), imgs.end());
    std::cout << "找到 " << imgs.size() << " 张图, 空格下一张, ESC 退出\n";

    for (size_t i = 0; i < imgs.size(); ++i) {
        cv::Mat img;
        std::vector<CoarseCorner> coarse;
        std::vector<cv::Point2f> refined;
        if (!processQrImage(imgs[i], img, coarse, refined)) {
            std::cerr << "处理失败: " << imgs[i] << "\n";
            continue;
        }
        cv::Mat vis = drawQrCorners(img, coarse, refined);
        std::cout << "[" << (i + 1) << "/" << imgs.size() << "] "
                  << fs::path(imgs[i]).filename().string()
                  << " (" << coarse.size() << " 个角点)\n";
        for (size_t j = 0; j < coarse.size(); ++j) {
            std::cout << "  corner[" << j << "]"
                      << "  YOLO(" << coarse[j].pt.x << "," << coarse[j].pt.y << ")"
                      << " w*h=" << coarse[j].wPx << "x" << coarse[j].hPx << "px"
                      << " -> Harris(" << refined[j].x << "," << refined[j].y << ")"
                      << "  d=" << cv::norm(refined[j] - coarse[j].pt) << " px\n";
        }
        cv::imshow("qr_corners", vis);
        int key = cv::waitKey(0) & 0xFF;
        if (key == 27) break;
    }
    cv::destroyAllWindows();
}

void saveLabelMeJson(const std::string& jsonPath,
                     const std::string& imageFilename,
                     int imgW, int imgH,
                     const std::vector<CoarseCorner>& coarse,
                     const std::vector<cv::Point2f>& refined) {
    std::ofstream f(jsonPath);
    if (!f) return;

    f << "{\n";
    f << "  \"version\": \"5.3.0\",\n";
    f << "  \"flags\": {},\n";
    f << "  \"shapes\": [\n";

    bool first = true;
    auto writeRect = [&](const std::string& label, int groupId,
                         float cx, float cy, float w, float h) {
        if (!first) f << ",\n";
        first = false;
        float hw = w * 0.5f, hh = h * 0.5f;
        f << "    {\n";
        f << "      \"label\": \"" << label << "\",\n";
        f << "      \"points\": [[" << (cx - hw) << ", " << (cy - hh)
          << "], [" << (cx + hw) << ", " << (cy + hh) << "]],\n";
        f << "      \"group_id\": " << groupId << ",\n";
        f << "      \"description\": \"\",\n";
        f << "      \"shape_type\": \"rectangle\",\n";
        f << "      \"flags\": {}\n";
        f << "    }";
    };

    const float refinedBoxSize = 6.0f;
    for (size_t i = 0; i < coarse.size(); ++i) {
        writeRect("coarse", (int)i,
                  coarse[i].pt.x, coarse[i].pt.y, coarse[i].wPx, coarse[i].hPx);
        writeRect("refined", (int)i,
                  refined[i].x, refined[i].y, refinedBoxSize, refinedBoxSize);
    }

    f << "\n  ],\n";
    f << "  \"imagePath\": \"" << imageFilename << "\",\n";
    f << "  \"imageData\": null,\n";
    f << "  \"imageHeight\": " << imgH << ",\n";
    f << "  \"imageWidth\": " << imgW << "\n";
    f << "}\n";
}

void saveQrCornerBatch(const std::string& inDir, const std::string& outDir) {
    namespace fs = std::filesystem;
    if (!fs::exists(inDir)) {
        std::cerr << "输入目录不存在: " << inDir << "\n";
        return;
    }
    fs::create_directories(outDir);

    std::vector<std::string> imgs;
    for (const auto& e : fs::directory_iterator(inDir)) {
        if (e.path().extension() == ".jpg") imgs.push_back(e.path().string());
    }
    std::sort(imgs.begin(), imgs.end());
    std::cout << "处理 " << imgs.size() << " 张图, 输出到: " << outDir << "\n";

    int savedCount = 0;
    for (size_t i = 0; i < imgs.size(); ++i) {
        cv::Mat img;
        std::vector<CoarseCorner> coarse;
        std::vector<cv::Point2f> refined;
        if (!processQrImage(imgs[i], img, coarse, refined)) {
            std::cerr << "处理失败: " << imgs[i] << "\n";
            continue;
        }

        std::string baseName = fs::path(imgs[i]).stem().string();
        std::string jpgPath  = outDir + "/" + baseName + ".jpg";
        std::string jsonPath = outDir + "/" + baseName + ".json";
        std::string visPath  = outDir + "/" + baseName + "_vis.jpg";

        cv::imwrite(jpgPath, img);
        saveLabelMeJson(jsonPath, baseName + ".jpg", img.cols, img.rows, coarse, refined);
        cv::imwrite(visPath, drawQrCorners(img, coarse, refined));

        std::cout << "[" << (i + 1) << "/" << imgs.size() << "] " << baseName
                  << " (" << coarse.size() << " 角点)\n";
        ++savedCount;
    }
    std::cout << "完成, 共保存 " << savedCount << " 张到 " << outDir << "\n";
}

}  // namespace barcode_detection
