#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "../dataset/dataset_handler.hh"

/**
 * Generate the image with predicted circles drawn and optionally ground truth bounding box
 *
 * @param image   Original Image with coins
 * @param circles Predicted circles coordinates and radius
 * @param labels  Ground Truth bounding boxes (optional)
 * @return        Image with the results and optionally ground truth boxes drawn
 */
cv::Mat draw_results(const cv::Mat& image, const cv::Mat& circles,
                     const std::vector<Label>& labels = {});

/**
 * Compute the performance of the coin detection algorithm with the EURO coins dataset
 *
 * @param dataset_path Path to the dataset root
 * @param save         Save the results images in the output directory. Defaults to false.
 */
void compute_metrics(const std::string& dataset_path, bool save = false);