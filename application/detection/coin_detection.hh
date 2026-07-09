#pragma once

#include <opencv2/opencv.hpp>

// Hyperparameters (calibrated experimentally on the EURO coins dataset)
extern const cv::Size GAUSSIAN_KERNEL_SIZE;
extern const int MIN_DIST;
extern const int MIN_RADIUS;
extern const int MAX_RADIUS;

/**
 * Apply preprocessing on the input image
 *
 * @param image The image we want to preprocess
 * @return      Image resulting from the preprocessing
 */
cv::Mat preprocess(const cv::Mat& image);

/**
 * Detect the circles in the image for coin detection
 *
 * @param image Image with coins
 * @return      Matrix (N x 3) of the form (center_x, center_y, radius),
 *              empty matrix if no circles are detected
 */
cv::Mat detect_circles(const cv::Mat& image);