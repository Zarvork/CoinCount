#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>

#include "feature_extraction.hh"
#include "../detection/coin_detection.hh"    
#include "../dataset/dataset_handler.hh"  


struct StandardScaler {
    cv::Mat mean;
    cv::Mat stddev;

    void fit(const cv::Mat& X);
    cv::Mat transform(const cv::Mat& X) const;
    cv::Mat fit_transform(const cv::Mat& X);
};

std::vector<std::vector<int>> stratified_k_fold_indices(const cv::Mat& y, int nSplits, unsigned int randomState = 42);

double compute_flatten_variance(const cv::Mat& X);

void build_fold_matrices(const cv::Mat& X, const cv::Mat& y,
                        const std::vector<int>& trainIdx, const std::vector<int>& valIdx,
                        cv::Mat& XTrain, cv::Mat& yTrain, cv::Mat& XVal, cv::Mat& yVal);

std::vector<int> complement_indices(const std::vector<std::vector<int>>& folds, int excluded);

double resolve_gamma(const std::string& gammaLabel, const cv::Mat& XScaledTrain);

cv::Ptr<cv::ml::SVM> train_SVM(const cv::Mat& XTrainScaled, const cv::Mat& yTrain, double C, double gamma);

struct CoinClassifier {
    StandardScaler scaler;
    cv::Ptr<cv::ml::SVM> svm;
    double bestC = 1.0;
    double bestGamma = 1.0;
    std::string bestGammaLabel;

    std::vector<int> predict(const cv::Mat& X) const;
    void save(const std::string& path) const;
    void load(const std::string& path);
};

CoinClassifier train_classifier(const cv::Mat& X, const cv::Mat& y);

// Resultat du pipeline complet de reconnaissance sur une image.
struct RecognitionResult {
    cv::Mat annotatedImage;              // image annotee avec cercles + valeurs predites
    std::vector<int> predictedClasses;   // classe predite pour chaque piece valide
    double totalValue = 0.0;             // somme totale en euros
};


cv::Mat draw_predictions(cv::Mat image, const std::vector<cv::Vec3f>& circles,
                          const std::vector<int>& validIndices,
                          const std::vector<int>& predictions);

RecognitionResult recognize_coins(const cv::Mat& image, const CoinClassifier& clf);