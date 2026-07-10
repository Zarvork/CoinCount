#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../dataset/dataset_handler.hh"
#include "../recognition/build_training_set.hh"

// Bounding box (top left and bottom right corners)
struct BoundingBox {
    double x1;
    double y1;
    double x2;
    double y2;
};

// Result of the IoU matching
struct MatchingResult {
    int tp;
    int fp;
    int fn;
    double iou_sum;
};

/**
 * Show an image
 *
 * @param img   The image we want to show
 * @param title Title of the image
 */
void show_image(const cv::Mat& img, const std::string& title);

/**
 * Compute IoU metric
 *
 * @param box_a Predicted bounding box
 * @param box_b Ground Truth bounding box
 * @return      IoU value
 */
double bb_intersection_over_union(const BoundingBox& box_a, const BoundingBox& box_b);

/**
 * Compute scoring metrics for the given predicted circles and ground truth boxes
 *
 * @param labels  Ground Truth bounding boxes
 * @param circles Predicted circles coordinates and radius
 * @return        Struct with True Positive, False Positive, False Negative and IoU sum metrics
 */
MatchingResult iou_matching(const std::vector<Label>& labels, const cv::Mat& circles);

/**
 * Plots the predicted circles with the class assigned to them by the IoU matching,
 * along with the ground truth bounding boxes, to visually verify that the matching is correct
 *
 * @param imaege  Original image containing the objects
 * @param labels  Ground truth bounding boxes
 * @param circles Predicted circles (center_x, center_y, radius)
 * @return        Annotated image with the predicted circles (predicted class in blue) and the ground truth boxes (true class in green)
 */
cv::Mat draw_matches(cv::Mat image, const std::vector<Label>& labels, const std::vector<cv::Vec3f>& circles);