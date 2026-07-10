#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

const int CROP_SIZE = 64;
const int HSV_N_BINS = 16;

const std::vector<std::pair<int, int>> LBP_SCALES = {
    {1, 8},
    {3, 24},
};

const int HOG_ORIENTATIONS = 8;
const int HOG_PIXELS_PER_CELL_ROWS = 8;
const int HOG_PIXELS_PER_CELL_COLS = 8;
const int HOG_CELLS_PER_BLOCK_ROWS = 2;
const int HOG_CELLS_PER_BLOCK_COLS = 2;

const int HOG_N_FEATURES = 1568; 
const int N_FEATURES = 1 + (6 + 3 * HSV_N_BINS) + 36 + HOG_N_FEATURES;


cv::Mat extract_coin_crop(const cv::Mat& image, float cx, float cy, float r);

cv::Mat extract_coin_mask(int size = CROP_SIZE);

std::vector<float> extract_size_feature(double r, int imageH, int imageW);

std::vector<float> extract_color_features(const cv::Mat& crop, const cv::Mat& mask = cv::Mat());

cv::Mat compute_uniform_lbp(const cv::Mat& gray, int nPoints, int radius);

std::vector<float> extract_lbp_features(const cv::Mat& crop, const cv::Mat& mask = cv::Mat());

std::vector<float> extract_hog_features(const cv::Mat& crop);

std::vector<float> extract_features(const cv::Mat& image, double cx, double cy, double r);

std::pair<cv::Mat, std::vector<int>> extract_features_batch(const cv::Mat& image, const std::vector<cv::Vec3f>& circles);

cv::Mat bgr_to_gray(const cv::Mat& image);

double bilinear_interpolate(const cv::Mat& gray, double row, double col);