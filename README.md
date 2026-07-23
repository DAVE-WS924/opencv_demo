# OpenCV 图像处理 Demo

一个用 C++ 和 OpenCV 实现的图像处理学习项目,把常见的图像处理算法按模块分类组织,通过命令行菜单交互。每个功能会弹出"原图 | 结果"并排窗口,方便对比。

## 功能菜单

```
===== 主菜单 =====
1. 滤波          - 均值 / 高斯 / 中值
2. 形态学        - 腐蚀 / 膨胀 / 开 / 闭 / 梯度
3. 阈值化        - 二值化 / 自适应 / Otsu
4. 直方图        - 计算 / 均衡化
5. 边缘检测      - Canny / Sobel
6. 特征检测      - Harris / Shi-Tomasi / ORB / SIFT / 特征匹配
7. 二维码角点    - YOLO + Harris 亚像素精定位
8. 轮廓检测      - findContours
```

## 目录结构

```
demo/
├── CMakeLists.txt
├── include/                     # 头文件 (声明)
│   ├── filter.h
│   ├── morphology.h
│   ├── thresholding.h
│   ├── histogram.h
│   ├── edge_detection.h
│   ├── feature_detection.h
│   ├── feature_point_detection.h
│   ├── barcode_detection.h
│   └── contour_detection.h
├── src/                         # 实现文件
│   ├── filter.cpp
│   ├── morphology.cpp
│   ├── thresholding.cpp
│   ├── histogram.cpp
│   ├── edge_detection.cpp
│   ├── feature_detection.cpp
│   ├── feature_point_detection.cpp
│   ├── barcode_detection.cpp
│   ├── contour_detection.cpp
│   └── run.cpp                  # 主程序 + 交互菜单
├── test.jpg / test1.jpg / test2.jpg   # 测试图
└── .gitignore
```

## 编译

依赖 OpenCV (在 3.4.13 上测试过)。

```bash
cd demo
mkdir build && cd build
cmake ..
make
```

## 运行

```bash
./build/run
```

启动后进入交互菜单:选分类 → 选具体算法 → 弹出"原图 | 结果"并排窗口,按任意键返回。

## 模块设计

每个分类对应一个 C++ namespace,头文件声明 + cpp 实现,互不耦合:

| namespace | 功能 | 主要函数 |
|---|---|---|
| `filter` | 滤波 | `blur`, `gaussianBlur`, `medianBlur` |
| `morphology` | 形态学 | `createKernel`, `erodeImage`, `dilateImage`, `morphOpen`, `morphClose`, `morphGradient` |
| `thresholding` | 阈值化 | `binarize`, `adaptiveBinarize`, `otsuBinarize` |
| `histogram` | 直方图 | `computeHist`, `equalizeHistogram`, `compareHistograms`, `drawHist` |
| `edge_detection` | 边缘检测 | `cannyEdge`, `sobelEdge` |
| `feature_detection` | 角点 | `harrisCornerDetection`, `detectHarrisCorners`, `shiTomasiCornerDetection`, `drawCorners` |
| `feature_point_detection` | 特征点 + 匹配 | `orbFeaturePointDetection`, `siftFeaturePointDetection`, `matchORBFeatures`, `drawFeatureMatches` |
| `barcode_detection` | 二维码角点 | `refineCornersWithHarris`, `runQrCornerBatch`, `saveQrCornerBatch` |
| `contour_detection` | 轮廓 | `detectContours` |

## 关键算法说明

### Harris 角点亚像素精定位 (`feature_detection::detectHarrisCorners`)

1. 计算全图 Harris 响应
2. 阈值筛选 + 非极大值抑制 (NMS)
3. 按响应强度保留 top-N
4. `cv::cornerSubPix` 二次曲面拟合,精度 0.05~0.1 px

### ORB 特征匹配 (`feature_point_detection::matchORBFeatures`)

1. ORB 检测关键点 + 计算二进制描述子
2. BFMatcher + Hamming 距离,KNN 匹配 (k=2)
3. Lowe's ratio test:`m[0].distance < 0.75 * m[1].distance` 过滤歧义匹配

### 二维码角点精定位 (`barcode_detection::refineCornersWithHarris`)

两阶段设计,把 YOLO 的粗角点 (±几像素误差) 精定位到亚像素:

- **阶段 1** (整数级): 在粗角点附近 searchRadius 半径内找 Harris 响应最强的像素
- **阶段 2** (亚像素): `cv::cornerSubPix` 用二次曲面拟合,精度到 0.05~0.1 px

## 依赖

- C++17
- OpenCV (tested on 3.4.13,兼容 4.x)
- CMake >= 3.10
