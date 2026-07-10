#include "coin_recognition.hh"
#include <algorithm>
#include <numeric>
#include <random>
#include <map>
#include <iostream>
#include <sstream>
#include <iomanip>

void StandardScaler::fit(const cv::Mat& X) {
    int nFeatures = X.cols;
    cv::reduce(X, mean, 0, cv::REDUCE_AVG, CV_32F);

    stddev = cv::Mat::zeros(1, nFeatures, CV_32F);
    for (int i = 0; i < X.rows; ++i) {
        cv::Mat diff = X.row(i) - mean;
        cv::Mat sq;
        cv::multiply(diff, diff, sq);
        stddev += sq;
    }
    stddev /= static_cast<float>(X.rows);
    cv::sqrt(stddev, stddev);

    for (int j = 0; j < nFeatures; ++j) {
        if (stddev.at<float>(0, j) < 1e-12f) {
            stddev.at<float>(0, j) = 1.0f;
        }
    }
}

cv::Mat StandardScaler::transform(const cv::Mat& X) const {
    cv::Mat result(X.rows, X.cols, CV_32F);
    for (int i = 0; i < X.rows; ++i) {
        cv::Mat row = (X.row(i) - mean) / stddev;
        row.copyTo(result.row(i));
    }
    return result;
}

cv::Mat StandardScaler::fit_transform(const cv::Mat& X) {
    fit(X);
    return transform(X);
}


std::vector<std::vector<int>> stratified_k_fold_indices(const cv::Mat& y, int nSplits, unsigned int randomState) {
    std::map<int, std::vector<int>> byClass;
    for (int i = 0; i < y.rows; ++i) {
        byClass[y.at<int>(i)].push_back(i);
    }

    std::mt19937 rng(randomState);
    for (auto& [classId, indices] : byClass) {
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    std::vector<std::vector<int>> folds(nSplits);
    for (auto& [classId, indices] : byClass) {
        for (size_t i = 0; i < indices.size(); ++i) {
            folds[i % nSplits].push_back(static_cast<int>(indices[i]));
        }
    }
    return folds;
}


double compute_flatten_variance(const cv::Mat& X) {
    cv::Scalar meanVal, stddevVal;
    cv::meanStdDev(X.reshape(1, 1), meanVal, stddevVal);
    return stddevVal[0] * stddevVal[0];
}

void build_fold_matrices(const cv::Mat& X, const cv::Mat& y,
                        const std::vector<int>& trainIdx, const std::vector<int>& valIdx,
                        cv::Mat& XTrain, cv::Mat& yTrain, cv::Mat& XVal, cv::Mat& yVal) {
    XTrain.create(static_cast<int>(trainIdx.size()), X.cols, CV_32F);
    yTrain.create(static_cast<int>(trainIdx.size()), 1, CV_32S);
    for (size_t i = 0; i < trainIdx.size(); ++i) {
        X.row(trainIdx[i]).copyTo(XTrain.row(static_cast<int>(i)));
        yTrain.at<int>(static_cast<int>(i)) = y.at<int>(trainIdx[i]);
    }

    XVal.create(static_cast<int>(valIdx.size()), X.cols, CV_32F);
    yVal.create(static_cast<int>(valIdx.size()), 1, CV_32S);
    for (size_t i = 0; i < valIdx.size(); ++i) {
        X.row(valIdx[i]).copyTo(XVal.row(static_cast<int>(i)));
        yVal.at<int>(static_cast<int>(i)) = y.at<int>(valIdx[i]);
    }
}

std::vector<int> complement_indices(const std::vector<std::vector<int>>& folds, int excluded) {
    std::vector<int> result;
    for (size_t f = 0; f < folds.size(); ++f) {
        if (static_cast<int>(f) == excluded) continue;
        result.insert(result.end(), folds[f].begin(), folds[f].end());
    }
    return result;
}

double resolve_gamma(const std::string& gammaLabel, const cv::Mat& XScaledTrain) {
    int nFeatures = XScaledTrain.cols;
    if (gammaLabel == "scale") {
        double variance = compute_flatten_variance(XScaledTrain);
        return 1.0 / (nFeatures * variance);
    }
    if (gammaLabel == "auto") {
        return 1.0 / nFeatures;
    }
    return std::stod(gammaLabel);
}

cv::Ptr<cv::ml::SVM> train_SVM(const cv::Mat& XTrainScaled, const cv::Mat& yTrain, double C, double gamma) {
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setC(C);
    svm->setGamma(gamma);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-6));
    svm->train(XTrainScaled, cv::ml::ROW_SAMPLE, yTrain);
    return svm;
}

std::vector<int> CoinClassifier::predict(const cv::Mat& X) const {
    cv::Mat Xscaled = scaler.transform(X);
    cv::Mat predictions;
    svm->predict(Xscaled, predictions);

    std::vector<int> result;
    result.reserve(predictions.rows);
    for (int i = 0; i < predictions.rows; ++i) {
        result.push_back(static_cast<int>(std::round(predictions.at<float>(i))));
    }
    return result;
}

void CoinClassifier::save(const std::string& path) const {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "mean" << scaler.mean;
    fs << "stddev" << scaler.stddev;
    fs << "bestC" << bestC;
    fs << "bestGamma" << bestGamma;
    fs << "bestGammaLabel" << bestGammaLabel;
    fs.release();
    svm->save(path + ".svm.xml");
}

void CoinClassifier::load(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    fs["mean"] >> scaler.mean;
    fs["stddev"] >> scaler.stddev;
    fs["bestC"] >> bestC;
    fs["bestGamma"] >> bestGamma;
    fs["bestGammaLabel"] >> bestGammaLabel;
    fs.release();
    svm = cv::ml::SVM::load(path + ".svm.xml");
}

CoinClassifier train_classifier(const cv::Mat& X, const cv::Mat& y) {
    const std::vector<double> Cs = {0.1, 1, 10, 100};
    const std::vector<std::string> gammaOptions = {"scale", "auto", "0.001", "0.01", "0.1"};
    const int nSplits = 5;

    auto folds = stratified_k_fold_indices(y, nSplits, 42);

    double bestScore = -1.0;
    double bestC = Cs[0];
    std::string bestGammaLabel = gammaOptions[0];

    for (double C : Cs) {
        for (const auto& gammaLabel : gammaOptions) {
            std::vector<double> foldScores;

            for (int k = 0; k < nSplits; ++k) {
                std::vector<int> trainIdx = complement_indices(folds, k);
                const std::vector<int>& valIdx = folds[k];

                cv::Mat XTrain, yTrain, XVal, yVal;
                build_fold_matrices(X, y, trainIdx, valIdx, XTrain, yTrain, XVal, yVal);

                StandardScaler scaler;
                cv::Mat XTrainScaled = scaler.fit_transform(XTrain);
                cv::Mat XValScaled   = scaler.transform(XVal);

                double gamma = resolve_gamma(gammaLabel, XTrainScaled);
                auto svm = train_SVM(XTrainScaled, yTrain, C, gamma);

                cv::Mat predictions;
                svm->predict(XValScaled, predictions);

                int correct = 0;
                for (int i = 0; i < predictions.rows; ++i) {
                    int pred = static_cast<int>(std::round(predictions.at<float>(i)));
                    if (pred == yVal.at<int>(i)) ++correct;
                }
                foldScores.push_back(static_cast<double>(correct) / predictions.rows);
            }

            double meanScore = std::accumulate(foldScores.begin(), foldScores.end(), 0.0) / foldScores.size();
            std::cout << "C=" << C << " gamma=" << gammaLabel << " -> accuracy=" << meanScore << std::endl;

            if (meanScore > bestScore) {
                bestScore = meanScore;
                bestC = C;
                bestGammaLabel = gammaLabel;
            }
        }
    }

    std::cout << "Best Hyperparameters : C=" << bestC << " gamma=" << bestGammaLabel << std::endl;
    std::cout << "Best accuracy : " << bestScore << std::endl;

    CoinClassifier clf;
    clf.scaler.fit(X);
    cv::Mat XScaled = clf.scaler.transform(X);
    double finalGamma = resolve_gamma(bestGammaLabel, XScaled);

    clf.svm = train_SVM(XScaled, y, bestC, finalGamma);
    clf.bestC = bestC;
    clf.bestGamma = finalGamma;
    clf.bestGammaLabel = bestGammaLabel;

    return clf;
}

cv::Mat draw_predictions(cv::Mat image, const std::vector<cv::Vec3f>& circles,
                          const std::vector<int>& validIndices,
                          const std::vector<int>& predictions) {
    for (size_t k = 0; k < validIndices.size(); ++k) {
        int idx = validIndices[k];
        int cx = static_cast<int>(circles[idx][0]);
        int cy = static_cast<int>(circles[idx][1]);
        int r  = static_cast<int>(circles[idx][2]);

        int classId = predictions[k];
        float value = CLASS_VALUES.at(classId);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << value << "E";

        cv::circle(image, cv::Point(cx, cy), r, cv::Scalar(255, 0, 0), 2);
        cv::putText(image, oss.str(),
            cv::Point(cx - r, cy + r + 18),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(255, 0, 0), 2);
    }
    return image;
}

RecognitionResult recognize_coins(const cv::Mat& image, const CoinClassifier& clf) {
    RecognitionResult result;
    result.annotatedImage = image.clone();

    cv::Mat circlesMat = detect_circles(image);
    if (circlesMat.empty()) {
        return result;
    }

    std::vector<cv::Vec3f> circles;
    if (circlesMat.channels() == 3) {
        circles.assign(circlesMat.begin<cv::Vec3f>(), circlesMat.end<cv::Vec3f>());
    } else {
        for (int i = 0; i < circlesMat.rows; ++i) {
            circles.emplace_back(
                circlesMat.at<float>(i, 0),
                circlesMat.at<float>(i, 1),
                circlesMat.at<float>(i, 2));
        }
    }
    if (circles.empty()) {
        return result;
    }

    auto [X, validIndices] = extract_features_batch(image, circles);
    if (X.rows == 0) {
        return result;
    }

    std::vector<int> predictions = clf.predict(X);

    result.predictedClasses = predictions;
    for (int classId : predictions) {
        result.totalValue += CLASS_VALUES.at(classId);
    }
    result.annotatedImage = draw_predictions(image.clone(), circles, validIndices, predictions);

    return result;
}