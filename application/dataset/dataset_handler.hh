#pragma once

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

// Dict to map dataset classes with euros values
extern const std::map<int, double> CLASS_VALUES;

struct Label {
    int class_id;
    int x1; // Top Left coordinate of bounding box
    int y1; // Top Left coordinate of bounding box
    int x2; // Bottom Right coordinate of bounding box
    int y2; // Bottom Right coordinate of bounding box
};

// A dataset sample
struct Sample {
    std::string filename;
    cv::Mat image;
    std::vector<Label> labels;
};

/**
 * Load a label file and returns a vector of Label with class_id, x1, y1, x2, y2
 * (top left and bottom right coordinates of bounding box)
 *
 * @param label_path Path to the label file containing image annotations in YOLO format
 * @param width      width of the dataset images
 * @param height     height of the dataset images
 * @return           Vector of Label with the class_id and bounding box coordinates
 */
std::vector<Label> load_labels(const std::string& label_path, int width, int height);

/**
 * Load the dataset into a vector of Sample
 *
 * @param dataset_path Path to the dataset root
 * @return             Vector of Sample (filename, image, labels)
 */
std::vector<Sample> load_dataset(const std::string& dataset_path);
