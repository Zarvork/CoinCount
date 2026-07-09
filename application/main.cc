#include <iostream>
#include <opencv2/opencv.hpp>
#include "detection/coin_detection.hh"

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    cv::Mat image = cv::imread("001.jpg");
    cv::Mat circles = detect_circles(image);
    std::cout << circles << std::endl;
    return 0;
}