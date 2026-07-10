#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "../recognition/coin_recognition.hh"

// Tests for StandardScaler

TEST(StandardScalerTest, MeanIsSubtractedCorrectly) {
    cv::Mat X = (cv::Mat_<float>(4, 2) << 1, 10, 2, 20, 3, 30, 4, 40);

    StandardScaler scaler;
    scaler.fit(X);

    EXPECT_NEAR(scaler.mean.at<float>(0, 0), 2.5f, 1e-5);
    EXPECT_NEAR(scaler.mean.at<float>(0, 1), 25.0f, 1e-5);
}

TEST(StandardScalerTest, PopulationStdDev) {
    cv::Mat X = (cv::Mat_<float>(4, 1) << 1, 2, 3, 4);

    StandardScaler scaler;
    scaler.fit(X);

    EXPECT_NEAR(scaler.stddev.at<float>(0, 0), std::sqrt(1.25f), 1e-5);
}

TEST(StandardScalerTest, TransformProducesZeroMeanUnitStd) {
    cv::Mat X = (cv::Mat_<float>(4, 2) << 1, 10, 2, 20, 3, 30, 4, 40);

    StandardScaler scaler;
    cv::Mat Xs = scaler.fit_transform(X);

    for (int j = 0; j < Xs.cols; ++j) {
        float sum = 0, sumSq = 0;
        for (int i = 0; i < Xs.rows; ++i) {
            sum += Xs.at<float>(i, j);
            sumSq += Xs.at<float>(i, j) * Xs.at<float>(i, j);
        }
        float mean = sum / Xs.rows;
        float var = sumSq / Xs.rows - mean * mean;
        EXPECT_NEAR(mean, 0.0f, 1e-4);
        EXPECT_NEAR(var, 1.0f, 1e-4);
    }
}

TEST(StandardScalerTest, ConstantColumnStddevSetToOne) {
    cv::Mat X = (cv::Mat_<float>(3, 1) << 5, 5, 5);

    StandardScaler scaler;
    cv::Mat Xs = scaler.fit_transform(X);

    EXPECT_FLOAT_EQ(scaler.stddev.at<float>(0, 0), 1.0f);
    for (int i = 0; i < Xs.rows; ++i) {
        EXPECT_TRUE(std::isfinite(Xs.at<float>(i, 0)));
        EXPECT_NEAR(Xs.at<float>(i, 0), 0.0f, 1e-6);
    }
}

TEST(StandardScalerTest, FitTransformEqualsFitThenTransform) {
    cv::Mat X = (cv::Mat_<float>(3, 2) << 1, 5, 2, 6, 3, 7);

    StandardScaler s1;
    cv::Mat a = s1.fit_transform(X);

    StandardScaler s2;
    s2.fit(X);
    cv::Mat b = s2.transform(X);

    ASSERT_EQ(a.size(), b.size());
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            EXPECT_NEAR(a.at<float>(i, j), b.at<float>(i, j), 1e-6);
}

// Tests for stratified_k_fold_indices

TEST(StratifiedKFoldTest, ReturnsRequestedNumberOfFolds) {
    cv::Mat y = (cv::Mat_<int>(10, 1) << 0,0,0,0,0,1,1,1,1,1);
    auto folds = stratified_k_fold_indices(y, 5);
    EXPECT_EQ(folds.size(), 5u);
}

TEST(StratifiedKFoldTest, IsAPartition) {
    cv::Mat y = (cv::Mat_<int>(10, 1) << 0,0,0,0,0,1,1,1,1,1);
    auto folds = stratified_k_fold_indices(y, 5);

    std::vector<int> all;
    for (const auto& fold : folds)
        for (int idx : fold)
            all.push_back(idx);

    std::sort(all.begin(), all.end());
    std::vector<int> expected(10);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(all, expected);
}

TEST(StratifiedKFoldTest, BalancedClassesEvenSplit) {
    cv::Mat y = (cv::Mat_<int>(10, 1) << 0,0,0,0,0,1,1,1,1,1);
    auto folds = stratified_k_fold_indices(y, 5);

    for (const auto& fold : folds) {
        int c0 = 0, c1 = 0;
        for (int idx : fold) {
            if (y.at<int>(idx) == 0) ++c0;
            else ++c1;
        }
        EXPECT_EQ(c0, 1);
        EXPECT_EQ(c1, 1);
    }
}

TEST(StratifiedKFoldTest, DeterministicWithSameSeed) {
    cv::Mat y = (cv::Mat_<int>(12, 1) << 0,0,0,0,0,0,1,1,1,1,1,1);
    auto a = stratified_k_fold_indices(y, 3, 42);
    auto b = stratified_k_fold_indices(y, 3, 42);
    EXPECT_EQ(a, b);
}

TEST(StratifiedKFoldTest, EachClassSpreadAcrossFolds) {
    cv::Mat y = (cv::Mat_<int>(12, 1) << 0,0,0,0,0,0,1,1,1,1,1,1);
    auto folds = stratified_k_fold_indices(y, 3);

    for (const auto& fold : folds) {
        std::set<int> classes;
        for (int idx : fold) classes.insert(y.at<int>(idx));
        EXPECT_EQ(classes.size(), 2u);
    }
}

// Tests for compute_flatten_variance

TEST(FlattenVarianceTest, KnownPopulationVariance) {
    cv::Mat X = (cv::Mat_<float>(2, 2) << 1, 2, 3, 4);
    EXPECT_NEAR(compute_flatten_variance(X), 1.25, 1e-5);
}

TEST(FlattenVarianceTest, ConstantMatrixHasZeroVariance) {
    cv::Mat X = (cv::Mat_<float>(2, 2) << 7, 7, 7, 7);
    EXPECT_NEAR(compute_flatten_variance(X), 0.0, 1e-9);
}

// Tests for resolve_gamma

TEST(ResolveGammaTest, AutoIsInverseNFeatures) {
    cv::Mat X = (cv::Mat_<float>(2, 4) << 1,2,3,4, 5,6,7,8);
    EXPECT_NEAR(resolve_gamma("auto", X), 0.25, 1e-9);
}

TEST(ResolveGammaTest, ScaleIsInverseNFeaturesTimesVariance) {
    cv::Mat X = (cv::Mat_<float>(2, 2) << 1, 2, 3, 4);
    double variance = compute_flatten_variance(X);
    double expected = 1.0 / (2 * variance);
    EXPECT_NEAR(resolve_gamma("scale", X), expected, 1e-9);
}

TEST(ResolveGammaTest, NumericStringIsParsed) {
    cv::Mat X = (cv::Mat_<float>(2, 2) << 1, 2, 3, 4);
    EXPECT_NEAR(resolve_gamma("0.01", X), 0.01, 1e-9);
    EXPECT_NEAR(resolve_gamma("0.1", X), 0.1, 1e-9);
}

// Tests for complement_indices

TEST(ComplementIndicesTest, ComplementIsUnionOfOtherFolds) {
    std::vector<std::vector<int>> folds = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
    };
    auto comp = complement_indices(folds, 1);

    std::set<int> comp_set(comp.begin(), comp.end());
    std::set<int> expected = {0, 1, 2, 6, 7, 8};
    EXPECT_EQ(comp_set, expected);
}

TEST(ComplementIndicesTest, ExcludedFoldNotInComplement) {
    std::vector<std::vector<int>> folds = {
        {0, 1}, {2, 3}, {4, 5},
    };
    auto comp = complement_indices(folds, 0);
    EXPECT_EQ(comp.size(), 4u);
    for (int idx : comp) {
        EXPECT_NE(idx, 0);
        EXPECT_NE(idx, 1);
    }
}

TEST(ComplementIndicesTest, ExcludingLastFold) {
    std::vector<std::vector<int>> folds = {
        {0, 1}, {2, 3}, {4, 5},
    };
    auto comp = complement_indices(folds, 2);
    std::set<int> comp_set(comp.begin(), comp.end());
    std::set<int> expected = {0, 1, 2, 3};
    EXPECT_EQ(comp_set, expected);
}

// Tests for build_fold_matrices

TEST(BuildFoldMatricesTest, CorrectShapes) {
    cv::Mat X = (cv::Mat_<float>(4, 2) << 1,1, 2,2, 3,3, 4,4);
    cv::Mat y = (cv::Mat_<int>(4, 1) << 0, 1, 0, 1);

    std::vector<int> trainIdx = {0, 2};
    std::vector<int> valIdx   = {1, 3};

    cv::Mat XTrain, yTrain, XVal, yVal;
    build_fold_matrices(X, y, trainIdx, valIdx, XTrain, yTrain, XVal, yVal);

    EXPECT_EQ(XTrain.rows, 2);
    EXPECT_EQ(XTrain.cols, 2);
    EXPECT_EQ(yTrain.rows, 2);
    EXPECT_EQ(XVal.rows, 2);
    EXPECT_EQ(yVal.rows, 2);
}

TEST(BuildFoldMatricesTest, CopiesCorrectRows) {
    cv::Mat X = (cv::Mat_<float>(4, 2) << 10,10, 20,20, 30,30, 40,40);
    cv::Mat y = (cv::Mat_<int>(4, 1) << 5, 6, 7, 8);

    std::vector<int> trainIdx = {0, 2};
    std::vector<int> valIdx   = {1, 3};

    cv::Mat XTrain, yTrain, XVal, yVal;
    build_fold_matrices(X, y, trainIdx, valIdx, XTrain, yTrain, XVal, yVal);

    EXPECT_FLOAT_EQ(XTrain.at<float>(0, 0), 10);
    EXPECT_FLOAT_EQ(XTrain.at<float>(1, 0), 30);
    EXPECT_EQ(yTrain.at<int>(0), 5);
    EXPECT_EQ(yTrain.at<int>(1), 7);

    EXPECT_FLOAT_EQ(XVal.at<float>(0, 0), 20);
    EXPECT_FLOAT_EQ(XVal.at<float>(1, 0), 40);
    EXPECT_EQ(yVal.at<int>(0), 6);
    EXPECT_EQ(yVal.at<int>(1), 8);
}