#include <iomanip>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "benchmark/benchmark.hh"
#include "detection/coin_detection.hh"
#include "utils/utils.hh"

// Print help message
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <command> [options]\n\n"
              << "Commands:\n"
              << "  benchmark  <dataset_path>              Compute detection and recognition metrics on the dataset\n"
              << "  detect     <dataset_path> <image_path> Detect coins on a single image\n"
              << "  recognize  <dataset_path> <image_path> Full pipeline on one image: detect + recognize + total\n"
              << "\n"
              << "Notes:\n"
              << "  <dataset_path> is always required (used to train the classifier for recognition).\n"
              << std::endl;
}

int main(int argc, char** argv) {
    // Verify the number of arguments
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];
    std::string dataset_path = argv[2];

    // Benchmark with detection + recognition metrics on the dataset
    if (command == "benchmark") {
        compute_metrics(dataset_path, true);
        return 0;
    }

    // Detect coins on an image
    if (command == "detect") {
        if (argc < 4) {
            std::cerr << "Error: 'detect' requires an image path.\n" << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        std::string image_path = argv[3];

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: could not load image " << image_path << std::endl;
            return 1;
        }

        // Coin Detection
        cv::Mat circles = detect_circles(image);
        if (circles.empty()) {
            std::cout << "No coins detected." << std::endl;
            return 0;
        }
        std::cout << "Coins detected: " << circles.rows << std::endl;

        // Draw and show the result
        cv::Mat result = draw_results(image, circles);
        cv::imwrite("result.jpg", result);
        show_image(result, "Detection");
        return 0;
    }

    // recognize : full pipeline on one image
    if (command == "recognize") {
        if (argc < 4) {
            std::cerr << "Error: 'recognize' requires an image path.\n" << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        std::string image_path = argv[3];

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: could not load image " << image_path << std::endl;
            return 1;
        }

        return 0;
    }

    // Unknown command
    std::cerr << "Error: unknown command '" << command << "'\n" << std::endl;
    print_usage(argv[0]);
    return 1;
}