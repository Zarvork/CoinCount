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

double compute_median(std::vector<double> values) { // copie volontaire (on va trier)
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());

    size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

void compute_metrics(const std::string& dataset_path, bool save) {
    // Load dataset
    std::vector<Sample> dataset = load_dataset(dataset_path);

    // Metric for detection
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

    // Metric for recognition
    std::cout << "=== Building of dataset ===" << std::endl;
    TrainingSet ts = build_training_set(true, dataset_path);
    cv::Mat X = ts.X;
    cv::Mat y = ts.y;

    std::cout << std::endl << "=== SVM Training ===" << std::endl;
    CoinClassifier clf = train_classifier(X, y);
    clf.save("coin_classifier.yml");
    std::cout << "Sauvegarde dans coin_classifier.yml" << std::endl;

    std::cout << std::endl << "=== Evaluation - Classification (5-fold cross-validation) ===" << std::endl;
    double accuracy;
    std::vector<int> yPred = evaluate_classification(X, y, clf.bestC, clf.bestGammaLabel, 5, accuracy);
    std::cout << "Accuracy : " << accuracy << " (" << accuracy * 100 << "%)" << std::endl;
    std::cout << "Objectif : > 80-85% -> " << (accuracy >= 0.80 ? "Succeeded" : "Fail") << std::endl;

    print_save_confusion_matrix(y, yPred);

    std::cout << std::endl << "=== Evaluation - Absolute Error for sum ===" << std::endl;

    auto [meanError, errors] = compute_sum_error(y, yPred, dataset_path);

    double medianError = compute_median(errors);
    double maxError = errors.empty() ? 0.0 : *std::max_element(errors.begin(), errors.end());

    std::cout << "Mean Absolute Error : " << std::fixed << std::setprecision(4) << meanError << "E" << std::endl;
    std::cout << "Median error : " << std::fixed << std::setprecision(4) << medianError << "E" << std::endl;
    std::cout << "Max error : " << std::fixed << std::setprecision(4) << maxError << "E" << std::endl;
    std::cout << "Objectif : < 0.50E -> " << (meanError < 0.50 ? "Succeeded" : "Fail") << std::endl;

}

std::vector<int> evaluate_classification(const cv::Mat& X, const cv::Mat& y,
                                         double C, const std::string& gammaLabel,
                                         int nSplits, double& accuracyOut) {
    auto folds = stratified_k_fold_indices(y, nSplits, 42);
    std::vector<int> yPred(y.rows, -1);

    for (int k = 0; k < nSplits; ++k) {
        std::vector<int> trainIdx;
        for (int f = 0; f < nSplits; ++f) {
            if (f == k) continue;
            trainIdx.insert(trainIdx.end(), folds[f].begin(), folds[f].end());
        }
        const std::vector<int>& valIdx = folds[k];

        cv::Mat XTrain(static_cast<int>(trainIdx.size()), X.cols, CV_32F);
        cv::Mat yTrain(static_cast<int>(trainIdx.size()), 1, CV_32S);
        for (size_t i = 0; i < trainIdx.size(); ++i) {
            X.row(trainIdx[i]).copyTo(XTrain.row(static_cast<int>(i)));
            yTrain.at<int>(static_cast<int>(i)) = y.at<int>(trainIdx[i]);
        }

        cv::Mat XVal(static_cast<int>(valIdx.size()), X.cols, CV_32F);
        for (size_t i = 0; i < valIdx.size(); ++i) {
            X.row(valIdx[i]).copyTo(XVal.row(static_cast<int>(i)));
        }

        StandardScaler scaler;
        cv::Mat XTrainScaled = scaler.fit_transform(XTrain);
        cv::Mat XValScaled   = scaler.transform(XVal);

        double gamma = resolve_gamma(gammaLabel, XTrainScaled);
        auto svm = train_SVM(XTrainScaled, yTrain, C, gamma);

        cv::Mat predictions;
        svm->predict(XValScaled, predictions);
        for (size_t i = 0; i < valIdx.size(); ++i) {
            yPred[valIdx[i]] = static_cast<int>(std::round(predictions.at<float>(static_cast<int>(i))));
        }
    }

    int correct = 0;
    for (int i = 0; i < y.rows; ++i) {
        if (yPred[i] == y.at<int>(i)) ++correct;
    }
    accuracyOut = static_cast<double>(correct) / y.rows;

    return yPred;
}

void print_save_confusion_matrix(const cv::Mat& y, const std::vector<int>& yPred,
                                  const std::string& csvPath) {
    std::vector<int> labels;
    for (const auto& [classId, value] : CLASS_VALUES) labels.push_back(classId);
    std::sort(labels.begin(), labels.end());

    int n = static_cast<int>(labels.size());
    std::map<int, int> labelToIdx;
    for (int i = 0; i < n; ++i) labelToIdx[labels[i]] = i;

    std::vector<std::vector<int>> cm(n, std::vector<int>(n, 0));
    for (int i = 0; i < y.rows; ++i) {
        int trueIdx = labelToIdx[y.at<int>(i)];
        int predIdx = labelToIdx[yPred[i]];
        cm[trueIdx][predIdx]++;
    }

    std::cout << std::endl << "Matrice de confusion :" << std::endl;
    std::cout << "vrai\\pred";
    for (int c : labels) std::cout << "\t" << CLASS_VALUES.at(c);
    std::cout << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << CLASS_VALUES.at(labels[i]);
        for (int j = 0; j < n; ++j) std::cout << "\t" << cm[i][j];
        std::cout << std::endl;
    }

    std::ofstream csv(csvPath);
    csv << "true_class,pred_class,count\n";
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            csv << CLASS_VALUES.at(labels[i]) << "," << CLASS_VALUES.at(labels[j]) << "," << cm[i][j] << "\n";

    std::cout << "Matrice de confusion sauvegardee dans " << csvPath << std::endl;
}

std::pair<double, std::vector<double>> compute_sum_error(
    const cv::Mat& y, const std::vector<int>& yPredCv, const std::string& datasetPath) {

    auto dataset = load_dataset(datasetPath);
    std::vector<double> errors;
    int sampleIdx = 0;

    for (const auto& [filename, image, labels] : dataset) {
        if (labels.empty()) continue;

        cv::Mat circlesMat = detect_circles(image);
        if (circlesMat.empty()) {
            double realSum = 0.0;
            for (const auto& lbl : labels) realSum += CLASS_VALUES.at(lbl.class_id);
            errors.push_back(realSum);
            continue;
        }

        std::vector<cv::Vec3f> circles;
        if (circlesMat.channels() == 3) {
            circles.assign(circlesMat.begin<cv::Vec3f>(), circlesMat.end<cv::Vec3f>());
        } else {
            for (int i = 0; i < circlesMat.rows; ++i) {
                circles.emplace_back(circlesMat.at<float>(i, 0), circlesMat.at<float>(i, 1), circlesMat.at<float>(i, 2));
            }
        }
        if (circles.empty()) continue;

        auto matches = match_circles_to_labels(labels, circles);
        if (matches.empty()) continue;

        int nValid = 0;
        for (const auto& m : matches) {
            cv::Mat crop = extract_coin_crop(image, m.cx, m.cy, m.r);
            if (!crop.empty()) ++nValid;
        }

        if (nValid == 0 || sampleIdx + nValid > static_cast<int>(yPredCv.size())) continue;

        double predictedSum = 0.0, realSum = 0.0;
        for (int i = 0; i < nValid; ++i) {
            predictedSum += CLASS_VALUES.at(yPredCv[sampleIdx + i]);
            realSum      += CLASS_VALUES.at(y.at<int>(sampleIdx + i));
        }
        sampleIdx += nValid;

        errors.push_back(std::abs(realSum - predictedSum));
    }

    double meanError = errors.empty() ? std::numeric_limits<double>::infinity()
                                       : std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    return { meanError, errors };
}