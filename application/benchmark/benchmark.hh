#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>

#include <opencv2/opencv.hpp>

#include "../dataset/dataset_handler.hh"
#include "../recognition/coin_recognition.hh"
#include "../recognition/build_training_set.hh"
#include "../dataset/dataset_handler.hh"
#include "../detection/coin_detection.hh"
#include "../recognition/feature_extraction.hh"


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
 * Compute the performance of the coin detection and recognition algorithm with the EURO coins dataset
 *
 * @param dataset_path Path to the dataset root
 * @param save         Save the results images in the output directory. Defaults to false.
 */
void compute_metrics(const std::string& dataset_path, bool save = false);

std::vector<int> evaluate_classification(const cv::Mat& X, const cv::Mat& y,
                                         double C, const std::string& gammaLabel,
                                         int nSplits, double& accuracyOut);

void print_save_confusion_matrix(const cv::Mat& y, const std::vector<int>& yPred,
                                  const std::string& csvPath = "confusion_matrix.csv");

std::pair<double, std::vector<double>> compute_sum_error(
    const cv::Mat& y, const std::vector<int>& yPredCv, const std::string& datasetPath);
