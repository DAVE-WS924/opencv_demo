cmake_minimum_required(VERSION 3.10)
project(opencv_learn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(run
    src/run.cpp
    src/filter.cpp
    src/morphology.cpp
    src/thresholding.cpp
    src/histogram.cpp
    src/edge_detection.cpp
    src/feature_detection.cpp
    src/barcode_detection.cpp
    src/contour_detection.cpp
    src/feature_point_detection.cpp
)
target_include_directories(run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(run PRIVATE ${OpenCV_LIBS})
