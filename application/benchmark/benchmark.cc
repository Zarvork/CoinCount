#include "benchmark.hh"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "../detection/coin_detection.hh"
#include "../dataset/dataset_handler.hh"
#include "../utils/utils.hh"

namespace fs = std::filesystem;

cv::Mat draw_results(const cv::Mat& image, const cv::Mat& circles,
                     const std::vector<Label>& labels) {
    cv::Mat result = image.clone();

    // Draw predicted circles
    for (int i = 0; i < circles.rows; ++i) {
        int cx = static_cast<int>(circles.at<float>(i, 0));
        int cy = static_cast<int>(circles.at<float>(i, 1));
        int r = static_cast<int>(circles.at<float>(i, 2));
        cv::circle(result, cv::Point(cx, cy), r, cv::Scalar(255, 0, 0), 2);
        cv::circle(result, cv::Point(cx, cy), 4, cv::Scalar(0, 0, 255), -1);
    }

    // Draw ground truth bounding box
    for (const Label& label : labels) {
        cv::rectangle(result,
                      cv::Point(label.x1, label.y1),
                      cv::Point(label.x2, label.y2),
                      cv::Scalar(0, 255, 0), 2);

        double value = CLASS_VALUES.at(label.class_id);
        std::ostringstream text;
        text << std::fixed << std::setprecision(2) << value << "E";
        cv::putText(result, text.str(),
                    cv::Point(label.x1, label.y1 - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
    }

    return result;
}

void compute_metrics(const std::string& dataset_path, bool save) {
    // Load dataset
    std::vector<Sample> dataset = load_dataset(dataset_path);

    int tp_total = 0;
    int fp_total = 0;
    int fn_total = 0;
    int no_detection = 0;
    double iou_total = 0;

    // Create directory to save results images
    if (save) {
        fs::create_directories("output");
    }

    // Iterate over each image of the dataset
    for (const Sample& sample : dataset) {
        const std::string& filename = sample.filename;
        const cv::Mat& image = sample.image;
        const std::vector<Label>& labels = sample.labels;

        // Coin Detection
        cv::Mat circles = detect_circles(image);

        int tp, fp, fn;
        double iou_sum;

        // Verify that circles are found
        if (!circles.empty()) {
            // Save images with predicted and ground truth boxes
            if (save) {
                cv::imwrite("output/" + filename, draw_results(image, circles, labels));
            }
            // Compute metrics for the current image
            MatchingResult res = iou_matching(labels, circles);
            tp = res.tp;
            fp = res.fp;
            fn = res.fn;
            iou_sum = res.iou_sum;
            if (fp > 0 || fn > 0) {
                std::cout << filename << ": TP=" << tp << " FP=" << fp
                          << " FN=" << fn << std::endl;
            }
        } else {
            tp = 0;
            fp = 0;
            fn = static_cast<int>(labels.size());
            iou_sum = 0;
            no_detection += 1;
        }

        tp_total += tp;
        fp_total += fp;
        fn_total += fn;
        iou_total += iou_sum;
    }

    double precision = (tp_total + fp_total) > 0
                           ? static_cast<double>(tp_total) / (tp_total + fp_total)
                           : 0;
    double recall = (tp_total + fn_total) > 0
                        ? static_cast<double>(tp_total) / (tp_total + fn_total)
                        : 0;
    double f1 = (precision + recall) > 0
                    ? (2 * precision * recall) / (precision + recall)
                    : 0;
    double mean_iou = tp_total > 0 ? iou_total / tp_total : 0;

    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Recall: " << recall << std::endl;
    std::cout << "f1: " << f1 << std::endl;
    std::cout << "Mean IoU: " << mean_iou << std::endl;
    std::cout << "Images without detection: " << no_detection << "/"
              << dataset.size() << std::endl;
}