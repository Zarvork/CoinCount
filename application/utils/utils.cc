#include "utils.hh"

#include <algorithm>

void show_image(const cv::Mat& img, const std::string& title) {
    cv::imshow(title, img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

double bb_intersection_over_union(const BoundingBox& box_a, const BoundingBox& box_b) {
    double x_a = std::max(box_a.x1, box_b.x1);
    double y_a = std::max(box_a.y1, box_b.y1);
    double x_b = std::min(box_a.x2, box_b.x2);
    double y_b = std::min(box_a.y2, box_b.y2);

    double inter_area = std::max(0.0, x_b - x_a) * std::max(0.0, y_b - y_a);
    double box_a_area = (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1);
    double box_b_area = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1);

    double iou = inter_area / (box_a_area + box_b_area - inter_area);

    return iou;
}

MatchingResult iou_matching(const std::vector<Label>& labels, const cv::Mat& circles) {
    // Convert circles into bounding box
    std::vector<BoundingBox> pred_boxes;
    for (int i = 0; i < circles.rows; ++i) {
        double cx = circles.at<float>(i, 0);
        double cy = circles.at<float>(i, 1);
        double r = circles.at<float>(i, 2);
        pred_boxes.push_back({cx - r, cy - r, cx + r, cy + r});
    }

    std::vector<BoundingBox> gt_boxes;
    for (const Label& label : labels) {
        gt_boxes.push_back({
            static_cast<double>(label.x1),
            static_cast<double>(label.y1),
            static_cast<double>(label.x2),
            static_cast<double>(label.y2),
        });
    }

    // Compute all IoU between all pairs of predicted and ground truth boxes
    // A pair is (iou, pred_index, gt_index)
    std::vector<std::tuple<double, int, int>> all_pairs;
    for (size_t i = 0; i < pred_boxes.size(); ++i) {
        for (size_t j = 0; j < gt_boxes.size(); ++j) {
            double iou = bb_intersection_over_union(pred_boxes[i], gt_boxes[j]);
            all_pairs.push_back({iou, static_cast<int>(i), static_cast<int>(j)});
        }
    }

    // Sort by descending order of IoU
    std::sort(all_pairs.begin(), all_pairs.end(),
              [](const auto& a, const auto& b) {
                  return std::get<0>(a) > std::get<0>(b);
              });

    std::set<int> matched_pred;
    std::set<int> matched_gt;
    int tp = 0;
    double iou_sum = 0;

    // Iterate over each pair (iou, pred_index, gt_index)
    for (const auto& [iou, i, j] : all_pairs) {
        if (iou < 0.5) {
            break;  // The remaining pairs all have an IoU < 0.5 (so no good matches left)
        }
        // Verify that the boxes are not already taken
        if (matched_pred.find(i) == matched_pred.end() &&
            matched_gt.find(j) == matched_gt.end()) {
            iou_sum += iou;
            tp += 1;
            matched_pred.insert(i);
            matched_gt.insert(j);
        }
    }

    int fp = static_cast<int>(pred_boxes.size()) - tp;  // Predicted boxes not assigned to any ground truth boxes
    int fn = static_cast<int>(gt_boxes.size()) - tp;    // Ground Truth boxes not assigned to any predicted boxes

    return {tp, fp, fn, iou_sum};
}