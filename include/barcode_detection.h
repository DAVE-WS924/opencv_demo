#pragma once
#include <opencv2/opencv.hpp>

namespace barcode_detection {
  
struct YoloMark {
    int cls;
    double cx, cy, w, h;  // 归一化
    double conf;          // 置信度, 不使用
};

// 读取 txt, 一行 = 一个角点
std::vector<YoloMark> readYoloMarks(const std::string& txtPath);
// 把 (cx, cy) 转成像素坐标的粗角点, w/h 也按比例换算到像素 (用于推算 ROI 半径)
struct CoarseCorner {
    cv::Point2f pt;   // 像素坐标
    float wPx, hPx;   // 该标注框在像素层面的 w, h
};
std::vector<CoarseCorner> marksToCoarse(const std::vector<YoloMark>& marks,
                                         int W, int H);

// 在每个粗角点附近开 ROI, 用 Harris 找该 ROI 内响应最强的点, 再做亚像素细化
//   searchRadius : 在粗角点周围多大半径内搜索 (像素), YOLO 精度高时调小可避免抓到相邻角
//   winSize      : 亚像素细化窗口
//   apertureSize : Sobel 算子大小, 3/5/7
//   k            : Harris 参数, 0.04~0.06
std::vector<cv::Point2f> refineCornersWithHarris(
    const cv::Mat& gray,
    const std::vector<CoarseCorner>& coarse,
    int searchRadius = 5,
    cv::Size winSize = cv::Size(5, 5),
    int blockSize = 2, int apertureSize = 3, double k = 0.04);
    

// 可视化: 橙色 = YOLO 粗点 (大空心圆), 红色 = Harris 精点 (小实心圆), 灰箭头连接
// 右上角加 4 倍局部放大图, 方便肉眼看亚像素偏移
cv::Mat drawQrCorners(const cv::Mat& img,
                      const std::vector<CoarseCorner>& coarse,
                      const std::vector<cv::Point2f>& refined);

// 处理一张图: 读 jpg + 同名 txt, txt 几行就精定位几个点
bool processQrImage(const std::string& imgPath,
                    cv::Mat& imgOut,
                    std::vector<CoarseCorner>& coarseOut,
                    std::vector<cv::Point2f>& refinedOut);

// 遍历 data/ 目录, 对每张图做精定位并依次显示
//   按空格/任意键下一张, ESC 退出
void runQrCornerBatch(const std::string& dir);

// 写 LabelMe 兼容的 JSON (X-anyLabeling 可直接打开)
// 8 个矩形: 4 个 label="coarse" (YOLO 原始位置, 用 YOLO 原 w/h) +
//           4 个 label="refined" (Harris 精定位, 用 6x6 小框)
// group_id = 角点序号 (0..3), 方便在 X-anyLabeling 里看配对关系
void saveLabelMeJson(const std::string& jsonPath,
                     const std::string& imageFilename,
                     int imgW, int imgH,
                     const std::vector<CoarseCorner>& coarse,
                     const std::vector<cv::Point2f>& refined);

// 批量保存: 处理 inDir 所有图, 输出到 outDir
//   <name>.jpg       原图 (X-anyLabeling 加载)
//   <name>.json      LabelMe JSON (8 矩形, coarse/refined 两个 label)
//   <name>_vis.jpg   可视化图 (橙圈 YOLO + 红点 Harris + 灰箭头)
void saveQrCornerBatch(const std::string& inDir, const std::string& outDir);

}