#include "dataset_handler.hh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <fstream>

// Dict to map dataset classes with euros values
const std::map<int, double> CLASS_VALUES = {
    {0, 0.01}, {1, 0.02}, {2, 0.05}, {3, 0.10},
    {4, 0.20}, {5, 0.50}, {6, 1.00}, {7, 2.00},
};

std::vector<Label> load_labels(const std::string& label_path, int width, int height) {
    std::vector<Label> labels;

    // Verify that the file exists
    if (!std::filesystem::exists(label_path)) {
        return labels;
    }

    // Read the file with image annotations in YOLO format
    std::ifstream file(label_path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        // Parse the line with class_id center_x center_y width height
        std::istringstream iss(line);
        int class_id;
        double center_x, center_y, box_width, box_height;
        if (!(iss >> class_id >> center_x >> center_y >> box_width >> box_height)) {
            continue;
        }

        // Compute top left coordinates of the bounding box (YOLO normalizes coordinates)
        double x1 = (center_x - box_width / 2) * width;
        double y1 = (center_y - box_height / 2) * height;

        // Compute bottom right coordinates of the bounding box (YOLO normalizes coordinates)
        double x2 = (center_x + box_width / 2) * width;
        double y2 = (center_y + box_height / 2) * height;

        // Convert the coordinates to int
        Label label;
        label.class_id = class_id;
        label.x1 = static_cast<int>(std::lround(x1));
        label.y1 = static_cast<int>(std::lround(y1));
        label.x2 = static_cast<int>(std::lround(x2));
        label.y2 = static_cast<int>(std::lround(y2));
        labels.push_back(label);
    }

    return labels;
}

std::vector<Sample> load_dataset(const std::string& dataset_path) {
    std::cout << "Dataset path: " << dataset_path << std::endl;

    // Define path for images and labels directories
    std::filesystem::path image_dir = std::filesystem::path(dataset_path) / "images";
    std::filesystem::path label_dir = std::filesystem::path(dataset_path) / "labels";

    // Collect image filenames
    std::vector<std::string> filenames;
    for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
        // Get the extension part of the filename
        std::string ext = entry.path().extension().string();

        // Convert the extension to lowercase
        for (char& c : ext) {
            c = std::tolower(static_cast<unsigned char>(c));
        }

        // Verify that the extension is valid
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            filenames.push_back(entry.path().filename().string());
        }
    }

    std::sort(filenames.begin(), filenames.end());

    // Creation of the dataset
    std::vector<Sample> dataset;
    // Iterate over each image filename of the dataset
    for (const auto& filename : filenames) {
        // Define path for current image and label files
        std::filesystem::path img_path = image_dir / filename;
        std::filesystem::path label_path = label_dir / (std::filesystem::path(filename).stem().string() + ".txt");

        // Load image
        cv::Mat image = cv::imread(img_path.string());
        if (image.empty()) {
            std::cout << "Warning: load failed for " << img_path.string() << std::endl;
            continue;
        }

        int h = image.rows;
        int w = image.cols;

        // Load labels
        std::vector<Label> labels = load_labels(label_path.string(), w, h);
        dataset.push_back({filename, image, labels});
    }

    std::cout << "Dataset loaded : " << dataset.size() << " images." << std::endl;
    return dataset;
}