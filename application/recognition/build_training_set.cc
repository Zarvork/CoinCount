#include "build_training_set.hh"

std::vector<Match> match_circles_to_labels(
    const std::vector<Label>& labels,
    const std::vector<cv::Vec3f>& circles) {
    std::vector<BoundingBox> pred_boxes;

    pred_boxes.reserve(circles.size());
    for (const auto& c : circles) {
        float cx = c[0], cy = c[1], r = c[2];
        pred_boxes.push_back({cx - r, cy - r, cx + r, cy + r});
    }

    std::vector<BoundingBox> gt_boxes;
    gt_boxes.reserve(labels.size());
    for (const auto& lbl : labels) {
        gt_boxes.push_back({(float)(lbl.x1), (float)(lbl.y1),
                              (float)(lbl.x2), (float)(lbl.y2)});
    }

    std::vector<Pair> all_pairs;
    all_pairs.reserve(pred_boxes.size() * gt_boxes.size());

    for (size_t i = 0; i < pred_boxes.size(); ++i) {
        for (size_t j = 0; j < gt_boxes.size(); ++j) {
            float iou = bb_intersection_over_union(pred_boxes[i], gt_boxes[j]);
            all_pairs.push_back({iou, i, j});
        }
    }

    std::sort(all_pairs.begin(), all_pairs.end(),
              [](const Pair& a, const Pair& b) { return a.iou > b.iou; });

    std::set<size_t> matched_pred;
    std::set<size_t> matched_gt;
    std::vector<Match> matches;

    for (const auto& p : all_pairs) {
        if (p.iou < IOU_THRESHOLD) {
            break;
        }
        if (matched_pred.count(p.i) == 0 && matched_gt.count(p.j) == 0) {
            matched_pred.insert(p.i);
            matched_gt.insert(p.j);

            int class_id = labels[p.j].class_id;
            const auto& c = circles[p.i];
            matches.push_back({class_id, c[0], c[1], c[2]});
        }
    }

    return matches;
}

cv::Mat augment_image(const cv::Mat& image, double angle) {
    int h = image.rows;
    int w = image.cols;
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(w / 2.0f, h / 2.0f), angle, 1.0);
    cv::Mat rotated;
    cv::warpAffine(image, rotated, M, cv::Size(w, h));
    return rotated;
}

TrainingSet build_training_set(bool augment, std::string dataset_path) {
    auto dataset = load_dataset(dataset_path);
    std::vector<std::vector<float>> X_rows;
    std::vector<int> y_vals;

    for (const auto& [filename, image, labels] : dataset) {
        cv::Mat circles_mat = detect_circles(image);
        if (circles_mat.empty() || labels.empty())
            continue;

        std::vector<cv::Vec3f> circles;
        if (circles_mat.channels() == 3) {
            circles.assign(
                circles_mat.begin<cv::Vec3f>(),
                circles_mat.end<cv::Vec3f>());
        } else {
            circles.reserve(circles_mat.rows);
            for (int i = 0; i < circles_mat.rows; ++i) {
                circles.emplace_back(
                    circles_mat.at<float>(i, 0),
                    circles_mat.at<float>(i, 1),
                    circles_mat.at<float>(i, 2));
            }
        }

        if (circles.empty())
            continue;

        auto matches = match_circles_to_labels(labels, circles);
        int h = image.rows;
        int w = image.cols;

        for (const auto& m : matches) {
            cv::Mat crop = extract_coin_crop(image, m.cx, m.cy, m.r);
            if (crop.empty())
                continue;

            cv::Mat mask = extract_coin_mask(crop.rows);
            auto size_feat  = extract_size_feature(m.r, h, w);
            auto color_feat = extract_color_features(crop, mask);

            const auto& angles = augment ? AUGMENT_ANGLES : std::vector<int>{0};

            for (int angle : angles) {
                cv::Mat rotated = (angle != 0) ? augment_image(crop, angle) : crop;

                auto lbp_feat = extract_lbp_features(rotated, mask);
                auto hog_feat = extract_hog_features(rotated);

                std::vector<float> feat;
                feat.insert(feat.end(), size_feat.begin(), size_feat.end());
                feat.insert(feat.end(), color_feat.begin(), color_feat.end());
                feat.insert(feat.end(), lbp_feat.begin(), lbp_feat.end());
                feat.insert(feat.end(), hog_feat.begin(), hog_feat.end());
                X_rows.push_back(feat);
                y_vals.push_back(m.class_id);
            }
        }
    }

    int n_samples  = (int)(X_rows.size());
    int n_features = n_samples > 0 ? (int)(X_rows[0].size()) : 0;

    cv::Mat X(n_samples, n_features, CV_32F);
    for (int i = 0; i < n_samples; ++i) {
        std::memcpy(X.ptr<float>(i), X_rows[i].data(), n_features * sizeof(float));
    }

    cv::Mat y(y_vals, true);

    int n_orig = augment ? n_samples / (int)(AUGMENT_ANGLES.size()) : n_samples;
    std::cout << "Training set built: got " << n_samples << " samples with "
              << n_orig << " originaux x " << AUGMENT_ANGLES.size()
              << " rotations" << std::endl;

    return {X, y};
}