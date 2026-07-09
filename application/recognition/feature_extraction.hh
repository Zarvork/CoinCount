#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

const int CROP_SIZE = 64;
const int HSV_N_BINS = 16;


cv::Mat extract_coin_crop(const cv::Mat& image, float cx, float cy, float r);

cv::Mat extract_coin_mask(int size = CROP_SIZE);

std::vector<float> extract_size_feature(double r, int imageH, int imageW);

cv::Mat extract_color_features(const cv::Mat& crop, const cv::Mat& mask = cv::Mat());
