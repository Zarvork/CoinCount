#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "../dataset/dataset_handler.hh"
#include "../utils/utils.hh"
#include "../detection/coin_detection.hh"
#include "feature_extraction.hh"
#include <algorithm>
#include <set>


const float IOU_THRESHOLD = 0.5f;

extern const std::map<int, double> CLASS_VALUES;

using Box = std::tuple<double, double, double, double>;

const std::vector<int> AUGMENT_ANGLES = {0, 90, 180, 270};

struct TrainingSet {
    cv::Mat X;  
    cv::Mat y;  
};

struct Match {
    int class_id;
    float cx;
    float cy;
    float r;
};

struct Pair { 
    float iou;
    size_t i;
    size_t j; 
};

std::vector<Match> match_circles_to_labels(
    const std::vector<Label>& labels,
    const std::vector<cv::Vec3f>& circles);

TrainingSet build_training_set(bool augment = true, std::string dataset_path = "");
